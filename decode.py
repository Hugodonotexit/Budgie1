"""CUDA-graph decoding: one captured graph per generated token instead of ~6000 separate kernel launches.

Decoding one token through the eager path is bound by Python launching thousands of tiny ops (the GPU
is ~85% idle). `FastDecoder` runs the same single-token forward with fixed-size state, so the whole
48-layer step can be captured once and replayed:

    * K/V live in ring buffers (`Ring`) instead of tensors grown by torch.cat. A token at absolute
      position p goes to slot p % cap; `kpos` holds the position stored in every slot (-1 = empty), and
      the attention mask is built from it, so slot order does not matter (rotary is applied before
      caching). A global layer's ring holds the whole context (`cap` >= position + margin); a windowed
      layer's holds exactly the `keep_past + 1` positions a query can still see.
    * Conv states, retention states and the n-gram history are static tensors updated in place.
    * The position is a device tensor incremented inside the graph.

The engine is a scratchpad owned by the model, not by a cache: `load` copies a cache's state in when a
cache first decodes, `release` copies it back (rebuilding the cache's dynamic K/V) when anything other
than a single-token step needs the cache again. Engines are keyed by (batch, cap) and reused across
generations, so a graph is captured once per size class, not once per prompt.

The fast path is used only for CUDA, no_grad, single-token steps on an unpadded batch. Anything else
(padded batches, beam search, inputs_embeds, CPU) takes the original eager path unchanged.
Set BUDGIE_FAST_DECODE=0 to turn it off.
"""

import os
import weakref
from collections import OrderedDict

import torch

from transformers.modeling_outputs import BaseModelOutputWithPast

from .cache import BudgieCache
from .patterns import AttentionPattern
from .retention import document_positions
from .rotary import RotaryEmbedding
from .sdpa import repeat_kv, sdpa

ENABLED = os.environ.get("BUDGIE_FAST_DECODE", "1") != "0"
GROUPED_ATTENTION = os.environ.get("BUDGIE_GROUPED_ATTENTION", "1") != "0"
MARGIN = 128       # head-room above the current position when picking a global ring size
MAX_ENGINES = 4    # engines (and their graphs) kept per model


def cap_for(n):
    """Smallest ring size class >= n: 1024, 1536, 2048, 3072, 4096, ... (at most 1.5x wasted attention)."""
    c = 1024
    while True:
        for cand in (c, c * 3 // 2):
            if cand >= n:
                return cand
        c *= 2


class Ring:
    def __init__(self, B, kv_heads, cap, head_dim, dtype, device):
        self.cap = cap
        self.k = torch.zeros(B, kv_heads, cap, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.kpos = torch.full((cap,), -1, dtype=torch.long, device=device)


class FastDecoder:
    def __init__(self, model, template, B, cap):
        cfg = model.config
        self.model, self.cfg, self.B, self.cap = model, cfg, B, cap
        self.owner = None
        self.pos_host = 0
        first = template.layers[0]
        dev = first.device
        self.keep = max(cfg.ngram_orders) - 1 if cfg.ngram_orders else 0

        # Static twins of the template cache's conv / retention states (same layout, zeros).
        self.ecache = BudgieCache(cfg)
        for el, tl in zip(self.ecache.layers, template.layers):
            el.device, el.dtype = tl.device, tl.dtype
            for s in range(tl.number_of_states):
                if tl.is_conv_states_initialized[s]:
                    el.conv_states[s] = torch.zeros_like(tl.conv_states[s])
                    el.is_conv_states_initialized[s] = True
                    el.has_previous_state[s] = True
                    el.conv_kernel_size[s] = tl.conv_kernel_size[s]
                if tl.is_recurrent_states_initialized[s]:
                    el.recurrent_states[s] = torch.zeros_like(tl.recurrent_states[s])
                    el.is_recurrent_states_initialized[s] = True

        self.rings = {}
        for i in range(cfg.num_hidden_layers):
            if cfg.kv_owner(i) is not None:
                continue
            spec = cfg.layer_spec(i)
            pattern = AttentionPattern.from_spec(spec)
            cap_i = cap if pattern.is_global else min(pattern.keep_past + 1, cap)
            dtype = template.layers[i].keys.dtype
            self.rings[i] = Ring(B, spec["kv_heads"], cap_i, cfg.head_dim, dtype, dev)

        self.ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.last_ids = template.last_ids.clone() if self.keep else None
        self.last_doc_pos = template.last_doc_pos.clone() if template.last_doc_pos is not None else None

        # Warm up on a side stream (lazy allocations, cuBLAS workspaces), then capture.
        with torch.no_grad():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.hidden = self._step()
        self._clear()

    # ------------------------------------------------------------------ the captured step

    def _step(self):
        m, cfg, ids = self.model, self.cfg, self.ids
        B = ids.shape[0]
        x = m.embed.adaptive(ids)
        if m.embed.fp32_residual:
            x = x.float()
        ng = m.embed.ngram
        if ng.orders:
            ctx = torch.cat([self.last_ids, ids], 1)
            for n, table, proj in zip(ng.orders, ng.tables, ng.proj):
                x = x + proj(table(ng.bucket_ids(ctx, n, self.keep, 1)).reshape(B, 1, -1))
            self.last_ids.copy_(ctx[:, -self.keep:])
        cos_sin = m.rotary(self.pos[None])
        cache = self.ecache

        doc_pos = None
        if m.branches is not None:
            doc_pos = document_positions(ids == cfg.bos_token_id, self.last_doc_pos, None)
            block_size = len(cfg.block_pattern)

        shared = {}
        for i, layer in enumerate(m.layers):
            if m.branches is not None:
                block, p = divmod(i, block_size)
                if p == 0:
                    block_input = x
                if p == m._g_pos:
                    x = x + m.branches[block](block_input, cache, doc_pos, None).to(x.dtype)
            owner = cfg.kv_owner(i)
            a = self._attention(layer.self_attn, i, layer.attn_norm(x), cos_sin, shared, owner)
            x = x + a
            x = x + layer.ffn(layer.ffn_norm(x), cache, None)
        if doc_pos is not None:
            self.last_doc_pos.copy_(doc_pos[:, -1])
        self.pos.add_(1)
        return m.norm(x)

    def _attention(self, attn, i, h, cos_sin, shared, owner):
        B = h.shape[0]
        cache = self.ecache
        q = attn._queries(h, cos_sin, cache, None)
        if owner is None:
            xkv = attn.kv_conv(h, cache, None)
            k = attn.k_proj(xkv).view(B, 1, attn.kv_heads, attn.head_dim)
            v = attn.v_proj(xkv).view(B, 1, attn.kv_heads, attn.head_dim)
            if attn.k_norm is not None:
                k = attn.k_norm(k)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            if attn.rope:
                k = RotaryEmbedding.apply(k, *cos_sin)
            ring = self.rings[i]
            slot = self.pos % ring.cap
            ring.k.index_copy_(2, slot, k)
            ring.v.index_copy_(2, slot, v)
            ring.kpos.index_copy_(0, slot, self.pos)
            shared[i] = ring
        else:
            ring = shared[owner]
        allow = attn.pattern.allowed(self.pos, ring.kpos) & (ring.kpos >= 0)[None]  # [1, cap]
        if GROUPED_ATTENTION and attn.sinks is not None:
            out = self._grouped_attention(attn, q, ring, allow)
        else:
            k, v = repeat_kv(ring.k, attn.n_rep), repeat_kv(ring.v, attn.n_rep)
            out = sdpa(q, k, v, attn.scale, attn.sinks, mask=allow[None, None])
        return attn.o_proj(out.transpose(1, 2).reshape(B, 1, attn.heads * attn.head_dim))

    @staticmethod
    def _grouped_attention(attn, q, ring, allow):
        """One query token against the ring, read once: the query heads of a K/V head are stacked as
        rows of a matmul (no head expansion, no padded copies of K/V), scores and softmax in fp32, the
        learned sink as one extra softmax slot with a zero value."""
        B, H, _, D = q.shape
        kvh, n_rep = attn.kv_heads, attn.n_rep
        qg = q.reshape(B, kvh, n_rep, D)
        scores = torch.bmm(qg.reshape(B * kvh, n_rep, D), ring.k.reshape(B * kvh, ring.cap, D).transpose(1, 2),
                           out_dtype=torch.float32).view(B, kvh, n_rep, ring.cap) * attn.scale
        scores = scores.masked_fill(~allow[None, None], float("-inf"))
        sink = attn.sinks.float().view(1, kvh, n_rep, 1).expand(B, kvh, n_rep, 1)
        p = torch.softmax(torch.cat([sink, scores], -1), -1)[..., 1:].to(q.dtype)
        out = torch.bmm(p.reshape(B * kvh, n_rep, ring.cap), ring.v.reshape(B * kvh, ring.cap, D))
        return out.view(B, H, 1, D)

    # ------------------------------------------------------------------ state in / out

    def _clear(self):
        for layer in self.ecache.layers:
            for t in list(layer.conv_states.values()) + list(layer.recurrent_states.values()):
                if t is not None:
                    t.zero_()
        for r in self.rings.values():
            r.kpos.fill_(-1)
        self.pos.zero_()
        self.ids.zero_()
        if self.last_ids is not None:
            self.last_ids.fill_(self.cfg.pad_token_id)
        if self.last_doc_pos is not None:
            self.last_doc_pos.zero_()

    @torch.no_grad()
    def load(self, cache):
        past = cache.get_seq_length()
        dev = self.pos.device
        for el, cl in zip(self.ecache.layers, cache.layers):
            for s in range(cl.number_of_states):
                if cl.is_conv_states_initialized[s]:
                    el.conv_states[s].copy_(cl.conv_states[s])
                if cl.is_recurrent_states_initialized[s]:
                    el.recurrent_states[s].copy_(cl.recurrent_states[s])
        for i, r in self.rings.items():
            layer = cache.layers[i]
            n = layer.keys.shape[2] if layer.keys.numel() else 0
            r.kpos.fill_(-1)
            if n:
                positions = torch.arange(past - n, past, device=dev)
                slots = positions % r.cap
                r.k.index_copy_(2, slots, layer.keys)
                r.v.index_copy_(2, slots, layer.values)
                r.kpos.index_copy_(0, slots, positions)
        if self.last_ids is not None:
            self.last_ids.copy_(cache.last_ids)
        if self.last_doc_pos is not None:
            self.last_doc_pos.copy_(cache.last_doc_pos)
        self.pos.fill_(past)
        self.pos_host = past
        self.owner = weakref.ref(cache)
        cache.fast_engine = self

    @torch.no_grad()
    def release(self):
        """Write the state back into the cache that owns it (if it is still alive)."""
        cache = self.owner() if self.owner is not None else None
        self.owner = None
        if cache is None:
            return
        past, dev = self.pos_host, self.pos.device
        for el, cl in zip(self.ecache.layers, cache.layers):
            for s in range(cl.number_of_states):
                if cl.is_conv_states_initialized[s]:
                    cl.conv_states[s].copy_(el.conv_states[s])
                if cl.is_recurrent_states_initialized[s]:
                    cl.recurrent_states[s].copy_(el.recurrent_states[s])
        for i, r in self.rings.items():
            layer = cache.layers[i]
            pattern = AttentionPattern.from_spec(self.cfg.layer_spec(i))
            n = past if pattern.is_global else min(past, pattern.keep_past)
            slots = torch.arange(past - n, past, device=dev) % r.cap
            layer.keys = r.k.index_select(2, slots)
            layer.values = r.v.index_select(2, slots)
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = past
        if self.last_ids is not None:
            cache.last_ids = self.last_ids.clone()
        if self.last_doc_pos is not None:
            cache.last_doc_pos = self.last_doc_pos.clone()
        cache.fast_engine = None

    @torch.no_grad()
    def decode(self, input_ids):
        self.ids.copy_(input_ids)
        self.graph.replay()
        self.pos_host += 1
        return self.hidden.clone()


# ---------------------------------------------------------------------- entry points used by the model

def _engine_for(model, cache, B, past):
    cap = cap_for(past + 1 + MARGIN)
    engines = model.__dict__.setdefault("_fast_engines", OrderedDict())
    key = (B, cap)
    eng = engines.pop(key, None)
    if eng is None:
        while len(engines) >= MAX_ENGINES:
            _, old = engines.popitem(last=False)
            old.release()
        eng = FastDecoder(model, cache, B, cap)
    engines[key] = eng  # most recently used last
    if eng.owner is not None and eng.owner() is not cache:
        eng.release()
    eng.load(cache)
    return eng


def try_fast_decode(model, cache, input_ids, attention_mask):
    """Hidden states [B, 1, d] for a single-token step through the graph, or None if this call must
    take the eager path (then the caller releases any engine state first)."""
    if not ENABLED or cache is None or input_ids is None or input_ids.shape[1] != 1:
        return None
    if not input_ids.is_cuda or torch.is_grad_enabled() or cache.no_fast:
        return None
    if model.record_branch_stats or model.branch_device is not None:
        return None
    eng = cache.fast_engine
    if eng is None:
        past = cache.get_seq_length()
        if past == 0 or cache.last_ids is None and model.config.ngram_orders:
            return None
        if attention_mask is not None and attention_mask.dim() == 2 and not bool(attention_mask.all()):
            cache.no_fast = True  # padded batch: the ring masks assume none
            return None
        eng = _engine_for(model, cache, input_ids.shape[0], past)
    elif eng.pos_host >= eng.cap:  # outgrew this size class
        past = eng.pos_host
        eng.release()
        eng = _engine_for(model, cache, input_ids.shape[0], past)
    return eng.decode(input_ids)
