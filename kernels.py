"""Optional fused kernels: Liger (RMSNorm, RoPE, cross entropy) and xformers (attention).

Everything here is off until switched on, and every call site keeps its plain PyTorch path, so a machine without
these packages (or a GPU a kernel does not support) runs the same model. Switch on with `configure(liger=True,
xformers=True)`, or per kernel (`configure(rms_norm=True)`), or with BUDGIE_KERNELS=liger,xformers in the environment.

    rms_norm   Liger RMSNorm: every RMSNorm / QKNorm.                                  (norms.py)
    rope       Liger RoPE: the rotary embedding of queries and keys.                   (rotary.py)
    loss       Liger fused linear cross entropy on every cluster of the adaptive       (head.py)
               softmax: summed for the training loss, per token for evaluation NLL.
               The logits are never built.
    embedding  Liger embedding lookup for the adaptive and n-gram tables.              (embeddings.py)
    attention  xformers memory-efficient attention (its CUTLASS kernel) in place of    (sdpa.py)
               the raw aten call behind the sink-merged window attention.

Liger's kernels are wrapped as `torch.library` custom ops (like causal-conv1d in convolution.py), so
torch.compile treats each as one opaque node instead of breaking the graph. Liger's RoPE rotates in place, so the
op rotates a copy; its RMSNorm runs with in_place=False, because the residual stream's gradient is shared.

Liger's other kernels have no counterpart in this architecture: the FFN is d -> w1 -> w2 -> w3 -> d with SiLU and a
centred dSiLU and no gate (no SwiGLU / GeGLU), there is no LayerNorm, no conditioning signal (modulated RMSNorm), no
standalone softmax (attention's is inside the fused kernel; the loss uses a logsumexp inside Liger's cross entropy),
and multi-token attention, sparsemax and mHC would each change the architecture (and the checkpoints). The int2 x int8
matmul (liger_kernel.ops.experimental.mm_int8int2) is forward-only (int8 activations times 2-bit ternary weights, int32
out, no scales, no backward), for BitNet-style models; this model trains fp16 weights, so it has nothing to feed it.
"""

import functools
import os

import torch
import torch.nn.functional as F
from transformers.utils import logging

logger = logging.get_logger(__name__)

try:
    from liger_kernel.ops import LigerEmbeddingFunction
    from liger_kernel.ops.rms_norm import _str_to_casting_mode, rms_norm_backward, rms_norm_forward
    from liger_kernel.ops.rope import rope_backward, rope_forward
    from liger_kernel.ops.utils import calculate_settings
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    LIGER_ERROR = None
except Exception as e:  # not installed, or built against another torch / triton
    LIGER_ERROR = f"{type(e).__name__}: {e}"

try:
    from xformers.ops import fmha
    from xformers.ops.fmha import attn_bias as _xbias
    from xformers.ops.fmha import cutlass as _cutlass
    XFORMERS_ERROR = None
except Exception as e:
    XFORMERS_ERROR = f"{type(e).__name__}: {e}"

GROUPS = {"liger": ("rms_norm", "rope", "loss", "embedding"), "xformers": ("attention",)}
FLAGS = {name: False for names in GROUPS.values() for name in names}
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
IGNORE = -100


def available(name):
    """Whether the package behind kernel `name` imported."""
    return (XFORMERS_ERROR if name == "attention" else LIGER_ERROR) is None


def configure(**switches):
    """Set kernels on or off: `liger=True`, `xformers=True`, or a kernel by name (`rms_norm`, `rope`, `loss`,
    `embedding`, `attention`). A kernel whose package is not usable stays off, with a warning. Returns the active kernels."""
    for key, on in switches.items():
        if key not in GROUPS and key not in FLAGS:
            raise ValueError(f"unknown kernel {key!r}; expected one of {sorted(GROUPS) + sorted(FLAGS)}")
        for name in GROUPS.get(key, (key,)):
            if on and not available(name):
                logger.warning(f"{name} kernel not enabled: {XFORMERS_ERROR if name == 'attention' else LIGER_ERROR}")
                on = False
            FLAGS[name] = bool(on)
    return active()


def active():
    return [name for name, on in FLAGS.items() if on]


def describe():
    on = active()
    return ("Liger " + ", ".join(n for n in GROUPS["liger"] if n in on) if any(n in on for n in GROUPS["liger"]) else "Liger off") + \
           "; " + ("xformers attention" if "attention" in on else "xformers off")


def use(name, x):
    """Whether kernel `name` handles tensor x (CUDA, a dtype the kernels take)."""
    return FLAGS[name] and x.is_cuda and x.dtype in _DTYPES


def _from_environment():
    tokens = [t.strip() for t in os.environ.get("BUDGIE_KERNELS", "").split(",") if t.strip()]
    if tokens:
        configure(**{t: True for t in tokens})


# --------------------------------------------------------------------------- RMSNorm

if LIGER_ERROR is None:
    _LLAMA = _str_to_casting_mode["llama"]

    @torch.library.custom_op("budgie::liger_rms_norm", mutates_args=())
    def _liger_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """x [..., d] -> (x / rms(x) * weight in x's dtype, the fp32 inverse rms of every row)."""
        y, _, rstd, _, _, _ = rms_norm_forward(x.contiguous(), weight, eps, 0.0, "llama", None)
        return y, rstd

    @_liger_rms_norm.register_fake
    def _(x, weight, eps):
        return torch.empty(x.shape, dtype=x.dtype, device=x.device), x.new_empty(x.numel() // x.shape[-1], dtype=torch.float32)

    @torch.library.custom_op("budgie::liger_rms_norm_backward", mutates_args=())
    def _liger_rms_norm_backward(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = x.shape[-1]
        block, warps = calculate_settings(d)
        # in_place=False: dy may be the same tensor the residual branch receives
        dx, dw = rms_norm_backward(dy.contiguous().view(-1, d), x.contiguous().view(-1, d), weight, rstd, 0.0, _LLAMA, block, warps, False, None)
        return dx.view(x.shape), dw

    @_liger_rms_norm_backward.register_fake
    def _(dy, x, weight, rstd):
        return torch.empty(x.shape, dtype=dy.dtype, device=x.device), torch.empty_like(weight)

    def _rms_setup(ctx, inputs, output):
        x, weight, _ = inputs
        ctx.save_for_backward(x, weight, output[1])

    def _rms_backward(ctx, dy, _drstd):
        x, weight, rstd = ctx.saved_tensors
        dx, dw = _liger_rms_norm_backward(dy, x, weight, rstd)
        return dx, dw, None

    _liger_rms_norm.register_autograd(_rms_backward, setup_context=_rms_setup)


def rms_norm(x, weight, eps):
    """x / rms(x) * weight, in x's dtype (normalised in fp32 and rounded once, after the scale)."""
    return _liger_rms_norm(x, weight, eps)[0]


# --------------------------------------------------------------------------- RoPE

if LIGER_ERROR is None:
    @torch.library.custom_op("budgie::liger_rope", mutates_args=())
    def _liger_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> torch.Tensor:
        """x [B, T, H, D] -> a rotated copy (inverse: rotated the other way, which is the gradient). cos, sin
        [B or 1, T, D], fp32. Liger's kernel works in place, hence the copy."""
        out = x.clone(memory_format=torch.contiguous_format)
        dummy = out.new_zeros(out.shape[0], out.shape[1], 1, out.shape[3])   # the kernel always rotates a key tensor too
        rotate = rope_backward if inverse else rope_forward
        rotated = rotate(out.transpose(1, 2), dummy.transpose(1, 2), cos, sin)[0]
        return rotated.transpose(1, 2).contiguous()

    @_liger_rope.register_fake
    def _(x, cos, sin, inverse):
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)

    def _rope_setup(ctx, inputs, output):
        _, cos, sin, inverse = inputs
        ctx.save_for_backward(cos, sin)
        ctx.inverse = inverse

    def _rope_backward(ctx, dy):
        cos, sin = ctx.saved_tensors
        return _liger_rope(dy, cos, sin, not ctx.inverse), None, None, None

    _liger_rope.register_autograd(_rope_backward, setup_context=_rope_setup)


def rope(x, cos, sin):
    """Rotary embedding of x [B, H, T, D] by cos, sin [B or 1, T, D] (half-split convention, fp32 maths): the same
    function as RotaryEmbedding.apply, returned in x's dtype."""
    return _liger_rope(x.transpose(1, 2), cos, sin, False).transpose(1, 2)


# --------------------------------------------------------------------------- loss

@functools.lru_cache(maxsize=None)
def _fused_loss_module(softcap, reduction):
    # fp32 accumulation of the weight gradient across Liger's token chunks. With "sum" the gradients it precomputes in
    # the logits' dtype are (p - onehot), of order 1, and the loss scale / token count come after, in backward
    return LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE, reduction=reduction, softcap=softcap, accum_dtype=torch.float32)


def linear_cross_entropy(h, weight, target, softcap, reduction="sum"):
    """Cross entropy of softcap-ed (h @ weight.T) against `target`, in fp32, without materialising the logits.
    h [N, d]; weight [V, d]; target [N], IGNORE rows count 0. "sum": the scalar sum over rows (the training loss);
    "none": the [N] per-row values (evaluation, no gradient)."""
    return _fused_loss_module(softcap or None, reduction)(weight, h, target)


# --------------------------------------------------------------------------- embedding

def embedding(ids, weight):
    """weight[ids] by Liger's embedding kernel. Its backward adds each row's gradient into a zeroed copy of the table
    (in the table's dtype) with atomics, so, unlike F.embedding's sorted fp32 accumulation, it is not deterministic."""
    return LigerEmbeddingFunction.apply(weight, ids)


# --------------------------------------------------------------------------- attention

@functools.lru_cache(maxsize=256)
def _window_bias(batch, length, window, device):
    """Causal attention over `batch` sequences of `length` tokens laid end to end, each token seeing itself and the
    window - 1 before it (token i sees keys i - window + 1 .. i of its own sequence). CUTLASS supports this mask
    in backward as well as forward."""
    return _xbias.BlockDiagonalCausalMask.from_seqlens([length] * batch, device=device).make_local_attention(window)


def _bias(q, k, window, mask_type):
    """The xformers bias for a call of window attention, and whether the batch is folded into one long sequence."""
    if window is not None:
        if mask_type != 1:
            raise NotImplementedError("a sliding window with a bottom-right-aligned mask")
        return _window_bias(q.shape[0], q.shape[1], int(window), q.device), True
    return (_xbias.LowerTriangularMask() if mask_type == 1 else _xbias.LowerTriangularFromBottomRightMask()), False


def attention_forward(q, k, v, scale, window, mask_type):
    """Causal attention from xformers' CUTLASS kernel. q [B, Tq, H, D], k, v [B, Tk, H, D]; mask_type 1: top-left
    causal (Tq == Tk), 2: bottom-right (the queries are the last Tq keys); window: keys per query or None.
    Returns (output [B, Tq, H, D], log-sum-exp [B, H, Tq] fp32)."""
    bias, fold = _bias(q, k, window, mask_type)
    B, Tq = q.shape[:2]
    if fold:
        q, k, v = (t.reshape(1, -1, *t.shape[2:]) for t in (q, k, v))
    out, lse = fmha.memory_efficient_attention_forward_requires_grad(q, k, v, attn_bias=bias, scale=scale, op=_cutlass.FwOp)
    return out.reshape(B, Tq, *out.shape[2:]), lse[:, :, :Tq]


def attention_backward(grad, q, k, v, out, lse, scale, window, mask_type):
    """Gradients (dq, dk, dv) of attention_forward given the output `out` and log-sum-exp `lse` [B, H, Tq] it
    was differentiated at (the caller may change both together, as sdpa.py does to add the sink)."""
    bias, fold = _bias(q, k, window, mask_type)
    shapes = q.shape, k.shape, v.shape
    lse = F.pad(lse, (0, (-lse.shape[-1]) % 32)).contiguous()    # the kernel keeps it padded to a multiple of 32
    if fold:
        grad, q, k, v, out = (t.reshape(1, -1, *t.shape[2:]) for t in (grad, q, k, v, out))
    grads = fmha.memory_efficient_attention_backward(grad, out, lse, q, k, v, attn_bias=bias, scale=scale, op=_cutlass.BwOp)
    return tuple(g.reshape(s) for g, s in zip(grads, shapes))


# --------------------------------------------------------------------------- start-up check

def _close(a, b, tol):
    return (a.float() - b.float()).abs().max().item() <= tol * max(1.0, b.float().abs().max().item())


def self_check(device):
    """Run each enabled kernel, forward and backward, on small tensors on `device` against its PyTorch
    equivalent. A kernel that raises or disagrees is switched off, with a warning. Returns {kernel: "ok" or why not}."""
    report = {}
    for name in active():
        try:
            with torch.random.fork_rng(devices=[device]):   # the caller's random state is left as it was
                torch.manual_seed(0)
                bad = _CHECKS[name](device)
        except Exception as e:  # a kernel this GPU / triton cannot run raises anything
            bad = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
        report[name] = bad or "ok"
        if bad:
            logger.warning(f"{name} kernel switched off: {bad}")
            FLAGS[name] = False
    return report


def _check_rms_norm(device):
    for x_dtype in (torch.float16, torch.float32):         # fp32: the residual stream, with fp16 weights
        x = torch.randn(3, 37, 256, device=device, dtype=x_dtype).requires_grad_()
        w = (1 + 0.1 * torch.randn(256, device=device, dtype=torch.float16)).requires_grad_()
        xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
        xn = xr.float() * torch.rsqrt(xr.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        ref = wr * xn.to(wr.dtype)
        got = rms_norm(x, w, 1e-5).to(w.dtype)
        g = torch.randn_like(ref)
        ref.backward(g), got.backward(g)
        for label, a, b in (("y", got, ref), ("dx", x.grad, xr.grad), ("dw", w.grad, wr.grad)):
            if not _close(a, b, 2e-2):
                return f"rms_norm {label} differs from PyTorch (x {x_dtype})"


def _check_rope(device):
    B, H, T, D = 2, 3, 41, 64
    x = torch.randn(B, T, H, D, device=device, dtype=torch.float16).transpose(1, 2).requires_grad_()
    freqs = torch.arange(T, device=device, dtype=torch.float32)[None, :, None] * torch.rand(D // 2, device=device)[None, None]
    cos, sin = torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()
    before = x.detach().clone()
    xr = x.detach().clone().requires_grad_()
    half = D // 2
    xf = xr.float()
    ref = (xf * cos[:, None] + torch.cat([-xf[..., half:], xf[..., :half]], -1) * sin[:, None]).to(torch.float16)
    got = rope(x, cos, sin)
    g = torch.randn_like(ref)
    ref.backward(g), got.backward(g)
    if not torch.equal(x.detach(), before):
        return "rope rotated its input in place"
    if not _close(got, ref, 2e-3):
        return "rope output differs from PyTorch"
    if not _close(x.grad, xr.grad, 5e-3):
        return "rope gradient differs from PyTorch"


def _check_loss(device):
    N, d, V, cap = 300, 128, 500, 50.0
    h = torch.randn(N, d, device=device, dtype=torch.float16).requires_grad_()
    w = (0.2 * torch.randn(V, d, device=device, dtype=torch.float16)).requires_grad_()
    target = torch.randint(0, V, (N,), device=device)
    target[::7] = IGNORE
    hr, wr = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    logits = cap * torch.tanh((hr @ wr.t()).float() / cap)
    keep = target != IGNORE
    ref = F.cross_entropy(logits[keep], target[keep], reduction="sum")
    got = linear_cross_entropy(h, w, target, cap)
    (ref * 4.0).backward(), (got * 4.0).backward()
    if not _close(got, ref, 2e-3):
        return f"fused loss {got.item():.5f} differs from PyTorch {ref.item():.5f}"
    for label, a, b in (("dh", h.grad, hr.grad), ("dw", w.grad, wr.grad)):
        if not _close(a, b, 1e-2):
            return f"fused loss {label} differs from PyTorch"
    with torch.no_grad():
        per_token = linear_cross_entropy(h.detach(), w.detach(), target, cap, "none")
    if not _close(per_token[keep], F.cross_entropy(logits[keep].detach(), target[keep], reduction="none"), 2e-3) or per_token[~keep].abs().max() != 0:
        return "per-token fused cross entropy differs from PyTorch"
    if not torch.allclose(per_token.sum(), got.detach(), rtol=1e-4):
        return "per-token values do not sum to the summed loss"


def _check_embedding(device):
    table = torch.randn(300, 96, device=device, dtype=torch.float16).requires_grad_()
    ref_table = table.detach().clone().requires_grad_()
    ids = (torch.rand(4, 50, device=device) ** 3 * 300).long().clamp_max(299)      # skewed: some rows repeat many times
    g = torch.randn(4, 50, 96, device=device, dtype=torch.float16)
    got, ref = embedding(ids, table), F.embedding(ids, ref_table)
    got.backward(g), ref.backward(g)
    if not torch.equal(got, ref):
        return "embedding lookup differs from F.embedding"
    if not _close(table.grad, ref_table.grad, 1e-2):
        return "embedding gradient differs from F.embedding"


def _check_attention(device):
    from .sdpa import sdpa_window                     # circular at import time, fine here
    B, H, T, D, W = 2, 4, 200, 64, 48
    sinks = torch.randn(H, device=device)
    for window, tail in ((None, False), (W, False), (None, True)):
        q, k, v = (torch.randn(B, H, T, D, device=device, dtype=torch.float16, requires_grad=True) for _ in range(3))
        g = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
        results = {}
        for backend in (False, True):
            FLAGS["attention"] = backend
            qq, kk, vv, ss = (t.detach().clone().requires_grad_() for t in (q, k, v, sinks))
            out = sdpa_window(qq, kk, vv, D ** -0.5, ss, window, tail=tail)
            out.backward(g)
            results[backend] = (out, qq.grad, kk.grad, vv.grad, ss.grad)
        FLAGS["attention"] = True
        for label, a, b in zip(("output", "dq", "dk", "dv", "dsink"), results[True], results[False]):
            if not _close(a, b, 1e-2):
                return f"xformers attention {label} differs from the aten kernel (window {window}, tail {tail})"


_CHECKS = {"rms_norm": _check_rms_norm, "rope": _check_rope, "loss": _check_loss, "embedding": _check_embedding, "attention": _check_attention}

_from_environment()
