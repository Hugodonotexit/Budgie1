"""Rotary position embeddings."""

import math

import torch
from torch import nn

from .kernels import rope, use


class RotaryEmbedding(nn.Module):
    """Rotary embeddings (half-split convention). It has no parameters or buffers on purpose: the
    frequencies are recomputed on every call, because a model.half() would round a stored
    inv_freq buffer, and at position 1000 a 5e-4 relative error in the highest frequency is half
    a radian of rotation.

    The angle position * inv_freq is formed in fp64 and reduced mod 2 pi before it is cast to fp32. In fp32 the
    product's rounding grows with the absolute position (the worst pair is off by 0.008 rad at 256k and 0.031 rad
    at 1M, independently in q and k), and casting an unreduced fp64 angle to fp32 brings the same error back. The
    angles are filled ANGLE_CHUNK positions at a time, so the fp64 temporaries stay small at any length.

    `scaling` (config.rope_extension, None = off) stretches the positions for a longer context:
      {"type": "linear", "factor": s}   every frequency / s (position interpolation)
      {"type": "yarn", "factor": s, "original_max_position_embeddings": L, "beta_fast": 32, "beta_slow": 1}
          YaRN (Peng et al. 2023): a dimension that turns more than beta_fast times over L keeps its frequency, one that
          turns less than beta_slow times is divided by s, linear in between; cos and sin are multiplied by
          0.1 ln(s) + 1 (the attention temperature, as in Hugging Face's implementation)."""

    ANGLE_CHUNK = 1 << 16

    def __init__(self, head_dim, theta, scaling=None):
        super().__init__()
        self.head_dim, self.theta, self.scaling = head_dim, theta, scaling

    def inv_freq(self, device):
        """(inverse frequencies [head_dim / 2] in fp64, cos/sin multiplier)."""
        exponent = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float64) / self.head_dim
        inv = 1.0 / (self.theta**exponent)
        sc = self.scaling
        if not sc:
            return inv, 1.0
        s = float(sc["factor"])
        if sc["type"] == "linear":
            return inv / s, 1.0
        if sc["type"] != "yarn":
            raise ValueError(f"rope_extension type {sc['type']!r}: expected 'linear' or 'yarn'")
        turns = sc["original_max_position_embeddings"] * inv / (2 * math.pi)     # full turns over the trained length
        fast, slow = float(sc.get("beta_fast", 32)), float(sc.get("beta_slow", 1))
        keep = ((turns - slow) / (fast - slow)).clamp(0.0, 1.0)                  # 1: keep the frequency, 0: divide by s
        return inv / s * (1 - keep) + inv * keep, 0.1 * math.log(s) + 1.0

    def forward(self, position_ids):
        """position_ids [batch, seq] -> (cos, sin), each [batch, seq, head_dim], fp32."""
        inv, mscale = self.inv_freq(position_ids.device)
        B, T = position_ids.shape
        half = inv.shape[0]
        cos = torch.empty(B, T, 2 * half, dtype=torch.float32, device=position_ids.device)
        sin = torch.empty_like(cos)
        for s in range(0, T, self.ANGLE_CHUNK):
            e = min(s + self.ANGLE_CHUNK, T)
            angle = (position_ids[:, s:e, None].double() * inv[None, None, :]).remainder(2 * math.pi).float()
            c, sn = angle.cos(), angle.sin()
            if mscale != 1.0:
                c, sn = c * mscale, sn * mscale
            cos[:, s:e, :half] = cos[:, s:e, half:] = c
            sin[:, s:e, :half] = sin[:, s:e, half:] = sn
        return cos, sin

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def apply(cls, x, cos, sin):
        """x: [batch, heads, seq, head_dim]. Rotates in fp32, returns x's dtype."""
        if use("rope", x):
            return rope(x, cos, sin)
        cos, sin = cos[:, None], sin[:, None]
        xf = x.float()
        return (xf * cos + cls.rotate_half(xf) * sin).to(x.dtype)
