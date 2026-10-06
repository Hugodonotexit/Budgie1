"""Activation checkpoints whose input lives in pinned host memory.

Ordinary per-layer checkpointing keeps one [B, T, d] input per layer on the GPU until backward: at T = 128k
that is 256 MiB x 48 layers = 12 GiB. Here the input is copied to pinned host RAM in forward and fetched back
in backward, where the function is run again with grad to get the gradients; the GPU holds nothing but the
(small) K/V that the layers share. The copies are on the current stream, in order with the compute (no
overlap): about 20 ms per 256 MiB each way over PCIe.
"""

import torch


class HostPool:
    """Pinned host buffers, reused between layers and steps (pinning a fresh 256 MiB buffer every layer would
    cost more than the copy). A buffer is taken in forward and given back in backward, so the pool settles at
    the number of checkpoints alive at the end of a forward pass."""

    def __init__(self):
        self.free = {}

    def take(self, like):
        stack = self.free.setdefault((tuple(like.shape), like.dtype), [])
        if stack:
            return stack.pop()
        return torch.empty(like.shape, dtype=like.dtype, device="cpu", pin_memory=torch.cuda.is_available())

    def give(self, buf):
        self.free.setdefault((tuple(buf.shape), buf.dtype), []).append(buf)

    def clear(self):
        """Forget every buffer (a new sequence length means new shapes; the old ones would stay pinned)."""
        self.free.clear()


class _Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, pool, x, *extras):
        ctx.fn, ctx.pool, ctx.device = fn, pool, x.device
        buf = pool.take(x)
        buf.copy_(x, non_blocking=True)
        ctx.buf = buf
        ctx.tensor_at = [i for i, t in enumerate(extras) if torch.is_tensor(t)]
        ctx.plain = [None if torch.is_tensor(t) else t for t in extras]
        ctx.save_for_backward(*[extras[i] for i in ctx.tensor_at])
        return fn(x, *extras)

    @staticmethod
    def backward(ctx, *grads):
        extras = list(ctx.plain)
        for i, t in zip(ctx.tensor_at, ctx.saved_tensors):
            extras[i] = t.detach().requires_grad_(t.requires_grad)
        x = ctx.buf.to(ctx.device, non_blocking=True)
        ctx.pool.give(ctx.buf)
        ctx.buf = None
        x.requires_grad_(True)
        with torch.enable_grad():
            out = ctx.fn(x, *extras)
        outs = out if isinstance(out, tuple) else (out,)
        pairs = [(o, g) for o, g in zip(outs, grads) if g is not None and o.requires_grad]
        torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        return (None, None, x.grad, *[e.grad if torch.is_tensor(e) else None for e in extras])


def offload_checkpoint(fn, pool, x, *extras):
    """fn(x, *extras) -> tensor or tuple of tensors, with x [B, T, d] saved in `pool` instead of on the GPU.
    The tensors in `extras` are kept on the GPU (small ones: K/V, positions, a recurrent state), and get
    their gradients back; anything else in `extras` is passed through. Parameters used inside fn receive
    their gradients in backward as usual. Outputs may be integer tensors (no gradient). Without autograd
    (eval, no_grad) fn is simply called."""
    if not torch.is_grad_enabled() or not x.requires_grad:
        return fn(x, *extras)
    return _Checkpoint.apply(fn, pool, x, *extras)
