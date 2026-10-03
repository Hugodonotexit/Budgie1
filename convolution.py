"""Depthwise causal convolution, with the causal-conv1d CUDA kernel when it is usable."""

import torch.nn.functional as F
import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.utils import logging

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function, causal_conv1d_fwd_function
    _CAUSAL_CONV1D_ERROR = None
except Exception as e:  # not installed, or its compiled extension was built against another torch
    causal_conv1d_fn = None
    _CAUSAL_CONV1D_ERROR = f"{type(e).__name__}: {e}"

logger = logging.get_logger(__name__)
if causal_conv1d_fn is None:
    logger.warning(f"causal-conv1d is not usable ({_CAUSAL_CONV1D_ERROR}); using the PyTorch fallback")


def _kernel_layout(t):
    """The kernel takes channel-first or channel-last [B, D, L]; anything else is made contiguous."""
    return t if t.stride(2) == 1 or t.stride(1) == 1 else t.contiguous()


if causal_conv1d_fn is not None:
    # The kernel wrapped as one functional custom op with its own backward, so torch.compile treats
    # it as a single opaque node and fuses the code around it. Called directly, causal-conv1d's
    # autograd.Function writes into a strided `out=` buffer, which torch.compile cannot trace: every
    # conv was a graph break.
    @torch.library.custom_op("budgie::causal_conv1d", mutates_args=())
    def _kernel_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return causal_conv1d_fwd_function(_kernel_layout(x), weight, None, None, None, None, False)

    @_kernel_conv.register_fake
    def _(x, weight):
        return torch.empty_like(_kernel_layout(x))

    @torch.library.custom_op("budgie::causal_conv1d_backward", mutates_args=())
    def _kernel_conv_backward(x: torch.Tensor, weight: torch.Tensor, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx, dweight, _, _ = causal_conv1d_bwd_function(_kernel_layout(x), weight, None, _kernel_layout(dout),
                                                        None, None, None, None, False, False)
        return dx, dweight

    @_kernel_conv_backward.register_fake
    def _(x, weight, dout):
        return torch.empty_like(_kernel_layout(x)), torch.empty_like(weight)

    def _setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    def _backward(ctx, dout):
        x, weight = ctx.saved_tensors
        return _kernel_conv_backward(x, weight, dout)

    _kernel_conv.register_autograd(_backward, setup_context=_setup_context)


def causal_conv(x, weight):
    """Depthwise causal conv: x [B, D, L], weight [D, W] -> [B, D, L], where
    out[t] = sum_j weight[:, j] * x[t - (W - 1) + j], so weight[:, -1] multiplies the current token
    and positions before the start count as zero. Same convention as causal-conv1d."""
    width = weight.shape[1]
    if causal_conv1d_fn is not None and x.is_cuda and 2 <= width <= 4 and x.dtype == weight.dtype:
        return _kernel_conv(x, weight)
    return F.conv1d(F.pad(x, (width - 1, 0)), weight[:, None, :], groups=x.shape[1])


def causal_conv_backend():
    """Which implementation `causal_conv` uses, for logging."""
    if causal_conv1d_fn is not None:
        return "causal-conv1d CUDA kernel"
    return f"PyTorch fallback (causal-conv1d unusable: {_CAUSAL_CONV1D_ERROR})"


class CausalConv(nn.Module):
    """Depthwise causal conv over the sequence axis, one filter per channel, no bias, no activation.
    Starts as the identity (only the current-token tap is 1).

    With a cache, its last (width - 1) inputs are kept in the layer's conv state `state_idx`, so
    decoding one token at a time gives exactly what the full-sequence forward gives."""

    def __init__(self, channels, width, layer_idx, state_idx):
        super().__init__()
        self.width, self.layer_idx, self.state_idx = width, layer_idx, state_idx
        self.weight = nn.Parameter(torch.zeros(channels, width))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weight.zero_()
        self.weight[:, -1] = 1.0

    def forward(self, x, cache: Cache | None = None, token_mask=None):
        """x: [B, T, D]. token_mask [B, T] zeroes padded positions first, so left padding sees
        the same zeros an unpadded sequence's start does."""
        if token_mask is not None:
            x = x * token_mask[..., None].to(x.dtype)
        xt = x.transpose(1, 2)  # [B, D, T], a channel-last view
        T = xt.shape[-1]
        if cache is not None:
            # Returns the cached last (width - 1) inputs followed by xt (zero-padded on the left
            # for a first call shorter than that), and stores the new tail.
            xt = cache.update_conv_state(xt, self.layer_idx, self.state_idx, conv_kernel_size=self.width - 1)
        return causal_conv(xt, self.weight)[..., -T:].transpose(1, 2)
