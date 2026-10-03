"""Rotary position embeddings."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary embeddings (half-split convention). It has no parameters or buffers on purpose: the
    frequencies are recomputed in fp32 on every call, because a model.half() would round a stored
    inv_freq buffer, and at position 1000 a 5e-4 relative error in the highest frequency is half
    a radian of rotation."""

    def __init__(self, head_dim, theta):
        super().__init__()
        self.head_dim, self.theta = head_dim, theta

    def forward(self, position_ids):
        """position_ids [batch, seq] -> (cos, sin), each [batch, seq, head_dim], fp32."""
        exponent = torch.arange(0, self.head_dim, 2, device=position_ids.device, dtype=torch.float32) / self.head_dim
        freqs = position_ids[:, :, None].float() * (1.0 / (self.theta**exponent))[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def apply(cls, x, cos, sin):
        """x: [batch, heads, seq, head_dim]. Rotates in fp32, returns x's dtype."""
        cos, sin = cos[:, None], sin[:, None]
        xf = x.float()
        return (xf * cos + cls.rotate_half(xf) * sin).to(x.dtype)
