"""One transformer layer."""

import functools

import torch
from transformers.modeling_layers import GradientCheckpointingLayer

from .attention import Attention
from .configuration_budgie import BudgieConfig
from .feedforward import FeedForward
from .norms import RMSNorm
from .chunked import AttentionHalf, FFNHalf, staged
from .offload import offload_checkpoint


class DecoderLayer(GradientCheckpointingLayer):
    """x += attention(norm(x)), then x += ffn(norm(x)). Returns (x, kv): the attention's K/V when
    this layer owns it, which the model hands to any layer that reads it (config.kv_share).

    Long-sequence mode (`set_offload`): the layer is checkpointed in two halves, attention and FFN, each with
    its input kept in pinned host memory (offload.py). Backward then holds the working set of one half, not the
    sum of both: about half the peak activation memory of whole-layer checkpointing, and no per-layer input on
    the GPU. Only for training without a cache; everything else takes the ordinary path."""

    def __init__(self, config: BudgieConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = FeedForward(config, layer_idx)
        self.offload, self.chunk = None, None
        self._attn_fn, self._ffn_fn = self._attn_half, self._ffn_half
        self._whole_compiled = None

    def forward(self, x, cos_sin, cache, shared_kv, token_mask, attention_mask, past):
        if self.offload is not None and cache is None and token_mask is None and attention_mask is None and past == 0:
            if torch.is_grad_enabled():
                return self._forward_offloaded(x, cos_sin, shared_kv)
            if self.chunk and x.shape[1] > self.chunk:       # evaluation of a long sequence: the chunked forward, no host copies
                out = AttentionHalf(self, cos_sin, self.chunk).forward(x, *(() if shared_kv is None else tuple(shared_kv)))
                return FFNHalf(self, self.chunk).forward(out[0])[0], (tuple(out[1:]) if len(out) > 1 else None)
        a, kv = self.self_attn(self.attn_norm(x), cos_sin, cache, shared_kv, token_mask, attention_mask, past)
        x = x + a
        return x + self.ffn(self.ffn_norm(x), cache, token_mask), kv

    # ------------------------------------------------------------------ long-sequence mode
    def _attn_half(self, x, k=None, v=None, k_pos=None, *, cos_sin=None):
        shared = None if k is None else (k, v, k_pos)
        a, kv = self.self_attn(self.attn_norm(x), cos_sin, None, shared, None, None, 0)
        x = x + a
        return (x,) if kv is None else (x, *kv)

    def _ffn_half(self, x):
        return x + self.ffn(self.ffn_norm(x), None, None)

    def _forward_offloaded(self, x, cos_sin, shared_kv):
        extras = () if shared_kv is None else tuple(shared_kv)  # (k, v, key positions) of the owner layer
        if self.chunk and x.shape[1] > self.chunk:              # the halves a chunk at a time, forward and backward (chunked.py)
            out = staged(AttentionHalf(self, cos_sin, self.chunk), self.offload, x, *extras)
            return staged(FFNHalf(self, self.chunk), self.offload, out[0])[0], (tuple(out[1:]) if len(out) > 1 else None)
        out = offload_checkpoint(functools.partial(self._attn_fn, cos_sin=cos_sin), self.offload, x, *extras)
        x1, kv = out[0], (tuple(out[1:]) if len(out) > 1 else None)
        return offload_checkpoint(self._ffn_fn, self.offload, x1), kv

    def set_offload(self, pool, compile_halves=False, chunk=None):
        """pool: a HostPool to switch long-sequence mode on, None to switch it off. With `compile_halves` the
        two halves are torch.compile'd (as the whole layer is in the ordinary mode); if the whole layer was
        compiled, that is set aside while the mode is on and restored when it is off. `chunk`: sequences longer
        than this many tokens are computed a chunk at a time (chunked.py); the halves are then not compiled."""
        self.offload, self.chunk = pool, (chunk if pool is not None else None)
        if pool is None:
            self._attn_fn, self._ffn_fn = self._attn_half, self._ffn_half
            if self._whole_compiled is not None:
                self._compiled_call_impl, self._whole_compiled = self._whole_compiled, None
            return
        if getattr(self, "_compiled_call_impl", None) is not None:
            self._whole_compiled, self._compiled_call_impl = self._compiled_call_impl, None
        if compile_halves:
            self._attn_fn, self._ffn_fn = torch.compile(self._attn_half), torch.compile(self._ffn_half)
