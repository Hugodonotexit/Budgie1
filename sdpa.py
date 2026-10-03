"""Scaled dot-product attention with a learned per-head sink."""

import torch
import torch.nn.functional as F

SINK_PAD = 8  # extra head-dim columns for the sink; 8 keeps the memory-efficient kernel's alignment


def repeat_kv(x, n_rep):
    """[batch, kv_heads, seq, dim] -> [batch, kv_heads * n_rep, seq, dim] (a stride-0 view for one kv head)."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    if h == 1:
        return x.expand(b, n_rep, s, d)
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def sdpa(q, k, v, scale, sinks=None, mask=None, causal=False):
    """torch SDPA with an optional learned sink logit per head.

    The sink is one extra softmax slot with logit sinks[h] and a zero value: it soaks up attention
    mass without contributing to the output. It is added as a key at index 0 that every query
    scores exactly sinks[h] against (q gets a constant 1 in an extra column, the sink key holds
    sinks[h] / scale in it, real keys hold 0 there) and a zero value row, so any SDPA kernel does it.

    q: [B, H, Tq, D]; k, v: [B, H, Tk, D]; mask: bool, True = may attend, broadcastable to
    [B, H, Tq, Tk]; causal: top-left causal (needs Tq == Tk)."""
    if sinks is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)
    B, H, Tq, D = q.shape
    one = q.new_zeros(B, H, Tq, SINK_PAD)
    one[..., 0] = 1
    qa = torch.cat([q, one], -1)
    ka = F.pad(k, (0, SINK_PAD))
    va = F.pad(v, (0, SINK_PAD))
    sink_key = F.pad((sinks / scale).to(k.dtype)[None, :, None, None].expand(B, H, 1, 1), (D, SINK_PAD - 1))
    ka = torch.cat([sink_key, ka], 2)
    va = torch.cat([va.new_zeros(B, H, 1, D + SINK_PAD), va], 2)
    if causal:
        # top-left causal with the sink as key 0: prepend a dummy query row so real query i sees the
        # sink and tokens 0..i (its output row is dropped)
        qa = torch.cat([qa.new_zeros(B, H, 1, D + SINK_PAD), qa], 2)
        return F.scaled_dot_product_attention(qa, ka, va, is_causal=True, scale=scale)[:, :, 1:, :D]
    if mask is not None:
        mask = torch.cat([mask.new_ones(*mask.shape[:-1], 1), mask], -1)
    return F.scaled_dot_product_attention(qa, ka, va, attn_mask=mask, scale=scale)[..., :D]


def window_attention_supported(q, sinks):
    """Whether sdpa_window can run: the CUDA memory-efficient kernel, with a sink, head_dim a multiple of 8."""
    return q.is_cuda and sinks is not None and q.shape[-1] % 8 == 0 and q.dtype in (torch.float16, torch.bfloat16, torch.float32)


def sdpa_window(q, k, v, scale, sinks, window):
    """Causal attention over tokens 0 .. T-1 in which token i sees keys i - window + 1 .. i (window
    None: all of 0 .. i), plus the learned sink. Same result as the sink-column trick in sdpa() with the
    equivalent mask, but it runs the memory-efficient kernel's own sliding-window mode: no mask, no
    padded head dim, and key blocks outside the window are skipped. Measured on a V100 (16 heads x 64,
    window 2048, forward + backward): 1.7x faster at T 2048, 2.9x at 4096, 2.6x at 16k.

    q, k, v: [B, H, T, D] (k, v already expanded to H heads); sinks: [H]."""
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))   # [B, T, H, D], the kernel's layout
    out, _ = _window_attention(qt, kt, vt, sinks, float(scale), window)
    return out.transpose(1, 2)


# How the sink is added: the kernel returns the output O over the real keys and its log-sum-exp l. The
# sink is one more softmax slot with logit s and a zero value, so the final output is O * exp(l - L),
# L = logaddexp(l, s). Backward runs the kernel's backward with the final output and L: the probabilities
# it rebuilds, exp(score - L), are then the true ones including the sink's share, which makes dq, dk, dv
# exact. The sink's own gradient is -sum_i P_sink,i * <dO_i, O_i> (its value row is zero).
# Both directions are custom ops so torch.compile treats each as one opaque node: traced directly, the
# aten backward op fails in torch 2.13's fake-tensor meta function (a `scale` argument mismatch).

@torch.library.custom_op("budgie::window_attention", mutates_args=())
def _window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sinks: torch.Tensor, scale: float,
                      window: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k, v: [B, T, H, D] contiguous. Returns (output [B, T, H, D], L [B, H, T] fp32)."""
    T = q.shape[1]
    out, lse, _, _, _, _ = torch.ops.aten._efficient_attention_forward(
        q, k, v, None, None, None, None, None, 0.0, 1, True, scale=scale, window_size=window)
    lse = lse[:, :, :T]
    total = torch.logaddexp(lse, sinks.float()[None, :, None])
    out = (out.float() * torch.exp(lse - total).transpose(1, 2)[..., None]).to(q.dtype)
    return out, total.contiguous()


@_window_attention.register_fake
def _(q, k, v, sinks, scale, window):
    B, T, H, _ = q.shape
    return torch.empty_like(q), q.new_empty(B, H, T, dtype=torch.float32)


@torch.library.custom_op("budgie::window_attention_backward", mutates_args=())
def _window_attention_backward(grad: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                               total: torch.Tensor, sinks: torch.Tensor, scale: float,
                               window: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = q.shape[1]
    grad = grad.contiguous()
    lse = F.pad(total, (0, (-T) % 32)).contiguous()          # the kernel keeps its log-sum-exp padded to 32
    unused = torch.zeros((), dtype=torch.int64)              # dropout seed / offset; no dropout
    dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
        grad, q, k, v, None, out, None, None, T, T, lse, 0.0, unused, unused, 1, False, scale=scale, window_size=window)
    dot = (grad.float() * out.float()).sum(-1).transpose(1, 2)                   # <dO_i, O_i>, [B, H, T]
    dsink = -(torch.exp(sinks.float()[None, :, None] - total) * dot).sum((0, 2)).to(sinks.dtype)
    return dq, dk, dv, dsink


@_window_attention_backward.register_fake
def _(grad, q, k, v, out, total, sinks, scale, window):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(sinks)


def _window_setup_context(ctx, inputs, output):
    q, k, v, sinks, scale, window = inputs
    ctx.save_for_backward(q, k, v, output[0], output[1], sinks)
    ctx.scale, ctx.window = scale, window


def _window_backward(ctx, grad_out, grad_total):
    q, k, v, out, total, sinks = ctx.saved_tensors
    dq, dk, dv, dsink = _window_attention_backward(grad_out, q, k, v, out, total, sinks, ctx.scale, ctx.window)
    return dq, dk, dv, dsink, None, None


_window_attention.register_autograd(_window_backward, setup_context=_window_setup_context)
