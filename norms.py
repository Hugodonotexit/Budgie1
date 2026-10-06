"""Normalization layers."""

import torch
from torch import nn

from .kernels import rms_norm, use


class RMSNorm(nn.Module):
    """Root-mean-square norm over the last dimension with a learned per-channel scale.

    Normalizes in fp32 and returns the weight's dtype, so an fp32 residual stream comes out fp16."""

    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        if use("rms_norm", x):
            return rms_norm(x, self.weight, self.eps).to(self.weight.dtype)   # Liger: one pass, one rounding
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(self.weight.dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class QKNorm(RMSNorm):
    """QK-norm: RMS-normalizes each head's query (or key) vector over head_dim, with one learned
    scale per head_dim channel shared by all heads. Applied to [batch, seq, heads, head_dim]
    before rotary embeddings, so every query and key has the same length whatever the activations
    do, and attention logits can only grow through the learned scales, never through the
    projections. One QKNorm for the queries and one for the keys in every attention layer."""
