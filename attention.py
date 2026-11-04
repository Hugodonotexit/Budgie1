"""The attention layer."""

import math

import torch
from torch import nn

from transformers.cache_utils import Cache

from .cache import KV_CONV, Q_CONV
from .configuration_budgie import BudgieConfig
from .convolution import CausalConv
from .norms import QKNorm
from .patterns import AttentionPattern
from .rotary import RotaryEmbedding
from .sdpa import repeat_kv
from .sparse import compress_kv, initial_state, sparse_attention


class Attention(nn.Module):
    """One attention layer: grouped-query, QK-norm, learned sinks, and the layer kind's pattern
    (local / dilated / global, see AttentionPattern).

    Q reads its own depthwise causal conv of the normed input and K/V read a second one. A reader
    layer (config.kv_share) has no K, V, K/V conv or k_norm: it attends over its owner's K/V.

    A global layer that config.sparse_spec names is compressed + selected (sparse.py) instead of dense; its only
    extra parameter is `pool_score` [kv_heads, head_dim], the per-head vector that weights keys within a block."""

    def __init__(self, config: BudgieConfig, layer_idx: int):
        super().__init__()
        spec = config.layer_spec(layer_idx)
        self.layer_idx = layer_idx
        self.owner = config.kv_owner(layer_idx) is None
        self.pattern = AttentionPattern.from_spec(spec)
        self.rope = spec["rope"]
        self.heads, self.head_dim, self.kv_heads = config.num_attention_heads, config.head_dim, spec["kv_heads"]
        self.n_rep = self.heads // self.kv_heads
        self.scale = self.head_dim**-0.5
        # config.global_logit_scale: past this length a global layer's queries are scaled up by ln(p + 1) / ln(L)
        self.logit_scale_length = None
        self.set_global_logit_scale(getattr(config, "global_logit_scale", None))
        d = config.hidden_size
        self.q_conv = CausalConv(d, spec["conv"], layer_idx, Q_CONV)
        self.q_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, d, bias=False)
        self.q_norm = QKNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.sinks = nn.Parameter(torch.zeros(self.heads)) if config.attention_sinks else None
        self.sparse = config.sparse_spec(layer_idx)
        if self.sparse is not None:
            self.pool_score = nn.Parameter(torch.zeros(self.kv_heads, self.head_dim))
        if self.owner:
            self.kv_conv = CausalConv(d, spec["conv"], layer_idx, KV_CONV)
            self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
            self.k_norm = QKNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None

    def set_global_logit_scale(self, spec):
        """spec: config.global_logit_scale ({"type": "log", "trained_length": L} or None). Only global layers use it."""
        self.logit_scale_length = int(spec["trained_length"]) if spec and self.pattern.is_global else None

    def query_positions(self, start, T, device):
        """The absolute positions _queries needs for the logit scale, or None when no query reaches the trained
        length (the scale is exactly 1 there, so it is skipped)."""
        L = self.logit_scale_length
        if L is None or start + T < L:
            return None
        return torch.arange(start, start + T, device=device)

    def _queries(self, x, cos_sin, cache, token_mask, positions=None):
        """positions: the queries' absolute positions [T] (a tensor, so decoding can pass its own), only read by a
        global layer with a logit scale; None = all below the trained length."""
        B, T, _ = x.shape
        q = self.q_proj(self.q_conv(x, cache, token_mask)).view(B, T, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = q.transpose(1, 2)
        if positions is not None and self.logit_scale_length is not None:
            factor = (torch.log1p(positions.float()) / math.log(self.logit_scale_length)).clamp_min(1.0)
            q = (q.float() * factor[:, None]).to(q.dtype)
        return RotaryEmbedding.apply(q, *cos_sin) if self.rope else q

    def _own_kv(self, x, cos_sin, cache, token_mask, past):
        """Owner layers: this layer's K and V for the new tokens, merged with the cache when there
        is one, and the absolute position of every key returned."""
        B, T, _ = x.shape
        xkv = self.kv_conv(x, cache, token_mask)
        k = self.k_proj(xkv).view(B, T, self.kv_heads, self.head_dim)
        v = self.v_proj(xkv).view(B, T, self.kv_heads, self.head_dim)
        if self.k_norm is not None:
            k = self.k_norm(k)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if self.rope:
            k = RotaryEmbedding.apply(k, *cos_sin)
        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)  # keeps only what a later token can still see
        return k, v, torch.arange(past + T - k.shape[2], past + T, device=x.device)

    def compress(self, k, v, key_ok=None):
        """Pooled K/V of this sparse layer (sparse.compress_kv)."""
        return compress_kv(k, v, self.pool_score, self.sparse["compress"], key_ok)

    def attend_sparse(self, q, k, v, p0, *, kc=None, vc=None, blk_ok=None, key_ok=None, q_ok=None, state=None, q_before=None):
        """Compressed + selected attention of queries q [B, H, T, D] at positions p0 .. over keys 0 .. p0 + T - 1
        (k, v not expanded). kc/vc/blk_ok: precomputed pooled K/V (computed here when None). Selection needs what
        the queries before p0 chose: `state` (from a cache) or `q_before`, the `group` queries just before p0.
        Returns (output [B, H, T, D], state)."""
        if kc is None:
            kc, vc, blk_ok = self.compress(k, v, key_ok)
        if state is None and q_before is not None:
            state = initial_state(q_before, p0, kc, blk_ok, self.scale, self.sparse)
        return sparse_attention(q, k, v, kc, vc, blk_ok, p0=p0, spec=self.sparse, sinks=self.sinks, scale=self.scale,
                                key_ok=key_ok, q_ok=q_ok, state=state)

    def forward(self, x, cos_sin, cache: Cache | None, shared_kv, token_mask, attention_mask, past):
        """Returns (output, kv): kv is (k, v, key_positions) for an owner layer, None for a reader.
        `past` is the number of tokens already cached; attention_mask the 2D padding mask or None."""
        B, T, _ = x.shape
        q = self._queries(x, cos_sin, cache, token_mask, self.query_positions(past, T, x.device))
        if self.owner:
            kv = self._own_kv(x, cos_sin, cache, token_mask, past)
            k, v, k_pos = kv
        else:
            kv = None
            k, v, k_pos = shared_kv
        if self.sparse is not None:
            if k.shape[2] != past + T:
                raise RuntimeError(f"layer {self.layer_idx}: sparse attention needs every key, got {k.shape[2]} of {past + T}")
            key_ok = attention_mask[:, k_pos].bool() if attention_mask is not None else None
            q_ok = token_mask.bool() if token_mask is not None else None
            state = cache.sparse_state.get(self.layer_idx) if cache is not None else None
            if past and state is None:
                raise RuntimeError(f"layer {self.layer_idx}: the cache has no sparse-attention state (it was cropped or built elsewhere)")
            out, state = self.attend_sparse(q, k, v, past, key_ok=key_ok, q_ok=q_ok, state=state)
            if cache is not None:
                cache.sparse_state[self.layer_idx] = state
            return self.o_proj(out.transpose(1, 2).reshape(B, T, self.heads * self.head_dim)), kv
        # Expanded here rather than by SDPA's enable_gqa: torch rejects that on the memory-efficient
        # kernel (the only fast one on V100) and silently falls back to the O(T^2) math kernel.
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        if past == 0 and attention_mask is None:  # tokens 0..T-1, nothing padded
            out = self.pattern.attend(q, k, v, self.sinks, self.scale)
        else:
            q_pos = torch.arange(past, past + T, device=x.device)
            key_ok = attention_mask[:, k_pos].bool() if attention_mask is not None else None
            out = self.pattern.attend_masked(q, k, v, q_pos, k_pos, key_ok, self.sinks, self.scale)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.heads * self.head_dim)), kv
