"""A fixed-decay retention branch (RetNet-style linear attention) that runs in parallel with a block's
local layers and writes one gated residual term just before the block's global layer.

Per head, with decay lambda = ln2 / half_life and gamma = exp(-lambda) (fixed, not learned):

    S_t = gamma * S_{t-1} + (1 - gamma) * k_t v_t^T           fp32, zeroed at a document start
    o_t = q_t^T S_t / (1 - gamma^{n_t})                       n_t = tokens since the document start, t included

Dividing by (1 - gamma^n) turns S into a weighted average, so its scale stays bounded at any position
(unnormalized it would reach about 1e5 * |v| at a 64k half-life, past fp16's range). q and k are L2-
normalized. `retention_reference` is the token-by-token fp64 oracle; `retention_chunked` computes the same
thing `chunk` tokens at a time with every exponent <= 0.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer

from .cache import BRANCH_CONV
from .configuration_budgie import BudgieConfig
from .convolution import CausalConv
from .norms import RMSNorm


def document_positions(is_bos, prev=None, valid=None):
    """1-based position of every token within its document: [B, T] int64.

    Counted from the last document start (`is_bos`) at or before the token, or from the start of the
    sequence if there has been none; `prev` [B] is the position of the last token of an earlier call
    (a cache), so a document continues across calls. Padding (`valid` False) is not counted and gets 0.
    Tokens j <= i are in the same document exactly when i - j < doc_pos[i]. With no ids at all
    (`inputs_embeds`), pass an all-False `is_bos`: the sequence is then one document."""
    valid = torch.ones_like(is_bos) if valid is None else valid.bool()
    count = valid.long().cumsum(1)  # real tokens so far in this call, 1-based
    last_start = torch.cummax(torch.where(is_bos & valid, count, torch.zeros_like(count)), dim=1).values
    prev = torch.zeros(is_bos.shape[0], dtype=torch.long, device=is_bos.device) if prev is None else prev.long()
    pos = torch.where(last_start > 0, count - last_start + 1, prev[:, None] + count)
    return torch.where(valid, pos, torch.zeros_like(pos))


def retention_reference(q, k, v, lam, doc_pos, S_init=None):
    """Token-by-token fp64 oracle. q, k: [B, T, H, dk]; v: [B, T, H, dv]; lam: [H]; doc_pos: [B, T].
    Returns (o [B, T, H, dv], S [B, H, dk, dv]). S is zeroed at every token with doc_pos == 1."""
    q, k, v, lam = (t.double() for t in (q, k, v, lam))
    B, T, H, dk = q.shape
    S = torch.zeros(B, H, dk, v.shape[-1], dtype=torch.float64, device=q.device) if S_init is None else S_init.double().clone()
    gamma, one_minus_gamma = torch.exp(-lam), -torch.expm1(-lam)
    out = []
    for t in range(T):
        S = torch.where((doc_pos[:, t] == 1)[:, None, None, None], torch.zeros_like(S), S)
        S = gamma[None, :, None, None] * S + one_minus_gamma[None, :, None, None] * k[:, t, :, :, None] * v[:, t, :, None, :]
        n = doc_pos[:, t].double().clamp_min(1)[:, None]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S) / (-torch.expm1(-lam[None, :] * n))[..., None])
    return torch.stack(out, 1), S


def _chunk_terms(q, k, v, dp, lam):
    """Everything about a batch of chunks that does not depend on the state carried into it. Shapes
    [B, n, L, H, d] for q, k, v and [B, n, L] for dp (n chunks of L tokens). Returns
        O_intra  [B, n, L, H, dv]  the within-chunk part of the output, not yet divided by 1 - gamma^n
        cross_w  [B, n, L, H]      weight of q_i^T S_prev: e^{-lam (i+1)}, or 0 for tokens outside the carried document
        denom    [B, n, L, H]      1 - gamma^{n_i}
        U        [B, n, H, dk, dv] what the chunk adds to the state
        a        [B, n, H]         factor the incoming state is multiplied by (0 if the chunk ends in a new document)."""
    L, dev, dt = q.shape[2], q.device, q.dtype
    one_minus_gamma = -torch.expm1(-lam)  # 1 - gamma, without cancellation
    i = torch.arange(L, device=dev)
    lag = i[:, None] - i[None, :]
    allowed = (lag >= 0) & (lag < dp[..., :, None])                                       # j <= i and the same document, [B, n, L, L]
    decay = torch.exp(-lam[:, None, None] * lag.clamp_min(0).to(dt))                      # [H, L, L]
    A = torch.einsum("bnihd,bnjhd->bnhij", q, k) * (decay * one_minus_gamma[:, None, None])[None, None] * allowed[:, :, None]
    O_intra = torch.einsum("bnhij,bnjhv->bnihv", A, v)
    cont = dp > (i + 1)                                                                   # in the document that spans into this chunk
    cross_w = torch.exp(-lam * (i + 1).to(dt)[:, None]) * cont[..., None]
    denom = -torch.expm1(-lam * dp.to(dt).clamp_min(1)[..., None])
    weight = one_minus_gamma * torch.exp(-lam * (L - 1 - i).to(dt)[:, None])              # [L, H]
    in_last_doc = (L - 1 - i) < dp[..., L - 1:L]                                          # [B, n, L]
    U = torch.einsum("bnjhd,bnjhv->bnhdv", k * (weight * in_last_doc[..., None])[..., None], v)
    a = torch.exp(-lam * L) * cont[:, :, L - 1, None]
    return O_intra, cross_w, denom, U, a


def _run_chunks(q, k, v, dp, lam, S):
    """Chunk terms for all n chunks at once, then the only sequential part: S_c = a_c S_{c-1} + U_c,
    one fused op per chunk. Returns (o [B, n*L, H, dv], S after the last chunk)."""
    O_intra, cross_w, denom, U, a = _chunk_terms(q, k, v, dp, lam)
    n = U.shape[1]
    a = a[..., None, None]
    prev = []
    for c in range(n):
        prev.append(S)
        S = torch.addcmul(U[:, c], a[:, c], S)
    prev = torch.stack(prev, 1)                                                           # state entering each chunk, [B, n, H, dk, dv]
    O = (O_intra + cross_w[..., None] * torch.einsum("bnihd,bnhdv->bnihv", q, prev)) / denom[..., None]
    return O.flatten(1, 2), S


def retention_chunked(q, k, v, lam, doc_pos, S_init=None, chunk=64):
    """The same recurrence, `chunk` tokens at a time; same arguments and results as retention_reference.

    Within a chunk (i, j = 0 .. L-1, every exponent <= 0):
        cross:  O_i  = e^{-lam (i+1)} q_i^T S_prev            only for tokens in the previous chunk's last document
        intra:  O_i += sum_{j<=i, same doc} (1-gamma) e^{-lam (i-j)} (q_i . k_j) v_j
        state:  S    = e^{-lam L} S_prev + sum_{j in the chunk's last doc} (1-gamma) e^{-lam (L-1-j)} k_j v_j^T
    S_prev is kept only if the chunk's last token is in that same document. Only the S recurrence runs
    chunk after chunk; every other term is computed for all chunks at once. The `T % chunk` tokens left
    over form one shorter chunk of their own -- never padded, so nothing spurious enters a carried state."""
    B, T, H, dk = q.shape
    dv, dt = v.shape[-1], q.dtype
    lam = lam.to(dt)
    S = torch.zeros(B, H, dk, dv, dtype=dt, device=q.device) if S_init is None else S_init.to(dt)
    full = T // chunk * chunk
    outs = []
    for lo, hi, L in ((0, full, chunk), (full, T, T - full)):
        if hi > lo:
            n = (hi - lo) // L
            o, S = _run_chunks(q[:, lo:hi].reshape(B, n, L, H, dk), k[:, lo:hi].reshape(B, n, L, H, dk), v[:, lo:hi].reshape(B, n, L, H, dv),
                               doc_pos[:, lo:hi].reshape(B, n, L), lam, S)
            outs.append(o)
    return torch.cat(outs, 1), S


# ---------------------------------------------------------------------------
# running the branch on another GPU
# ---------------------------------------------------------------------------

_side_streams = {}


def _side_stream(device):
    if device not in _side_streams:
        _side_streams[device] = torch.cuda.Stream(device)
    return _side_streams[device]


def _hop(x, device, host_wait=False):
    """x copied to `device` through pinned host memory, on side streams of both GPUs.

    Only the destination waits for the source. Tensor.to() between GPUs also makes the SOURCE's stream
    wait for whatever the destination has queued, which would tie the two GPUs' streams together and
    remove the overlap this exists for. The result is ready on the destination's current stream.

    host_wait: block this thread until the copy has left the source before queueing the destination's
    wait. Used by the backward copies from the branch GPU, which run in autograd's worker thread for
    that GPU: without it the wait is queued when the branch's backward has just been queued, early in
    the main GPU's stream, which then stalls there until the branch's backward is done."""
    src_side, dst_side = _side_stream(x.device), _side_stream(device)
    src_side.wait_stream(torch.cuda.current_stream(x.device))
    host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
    with torch.cuda.stream(src_side):
        host.copy_(x, non_blocking=True)
    x.record_stream(src_side)
    left = src_side.record_event()
    if host_wait:
        left.synchronize()
    dst_side.wait_event(left)
    with torch.cuda.stream(dst_side):
        out = host.to(device, non_blocking=True)
    current = torch.cuda.current_stream(device)
    current.wait_stream(dst_side)
    out.record_stream(current)
    return out


class _ToDevice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, device, host_wait_back):
        ctx.src, ctx.host_wait_back = x.device, host_wait_back
        return _hop(x, device)

    @staticmethod
    def backward(ctx, grad):
        return _hop(grad, ctx.src, ctx.host_wait_back), None, None


def to_device(x, device, host_wait_back=False):
    """Differentiable copy of x to another GPU with one-way ordering (see _hop). host_wait_back: its
    gradient's copy back waits on the host (for copies onto the branch GPU; see _hop)."""
    if x.device == device:
        return x
    return _ToDevice.apply(x, device, host_wait_back)


def branch_half_lives(config: BudgieConfig, block: int):
    """The decay half-life (tokens) of each head of `block`'s branch, a list of `lin_heads` floats.

    One grid: head j of H sits at fraction (j + block / num_blocks) / H of the way (in log2) across
    `lin_half_life_range`, so the blocks, staggered, together cover the range more densely.

    A widened branch (`lin_base_heads` = B < H, see grow_branch.py): heads 0..B-1 stay exactly where the
    B-head grid put them, because their weights were trained for those decays. Head B + k, with
    j = k % B and m = 1 + k // B, goes m / H past old head j (wrapping around the range), which makes the
    union of the two sets one even grid of H points spaced 1/H apart."""
    lo, hi = config.lin_half_life_range
    H, base, off = config.lin_heads, config.lin_base_heads or config.lin_heads, block / config.num_blocks
    log_lo, span = math.log2(lo), math.log2(hi) - math.log2(lo)
    out = [2.0 ** (log_lo + span * (j + off) / base) for j in range(base)]
    for k in range(H - base):
        j, m = k % base, 1 + k // base
        frac = ((j + off) / base + m / H) % 1.0
        out.append(2.0 ** (log_lo + span * frac))
    return out


class RetentionBranch(GradientCheckpointingLayer):
    """The retention branch of one block: y = g * Wo(head_norm(retention(conv(norm(x0))))).

    With `lin_head_gate` each head's normalized output is first multiplied by a per-token sigmoid
    gate, sigmoid(W_h u + b_h), read from the same input u = conv(norm(x0)) as q, k and v: the branch
    can switch a head off where its long memory is no use. Initially W is small and b = 0, so every
    gate is about 0.5 (on top of g).

    Reads x0, the residual stream at the start of the block, through its own RMSNorm and causal conv
    (identity at init). No RoPE: the decay supplies recency. Everything from the L2-norm to the head
    norm is fp32 whatever the model dtype, and the state is always fp32; only the four projections
    run in the model dtype. Returns the residual term in fp32 [B, T, d]."""

    def __init__(self, config: BudgieConfig, block: int, layer_idx: int):
        super().__init__()
        d, H, dk = config.hidden_size, config.lin_heads, config.lin_head_dim
        self.heads, self.head_dim, self.chunk, self.layer_idx, self.eps = H, dk, config.lin_chunk, layer_idx, config.rms_norm_eps
        self.norm = RMSNorm(d, config.rms_norm_eps)
        self.conv = CausalConv(d, config.lin_conv_kernel, layer_idx, BRANCH_CONV)
        self.q_proj = nn.Linear(d, H * dk, bias=False)
        self.k_proj = nn.Linear(d, H * dk, bias=False)
        self.v_proj = nn.Linear(d, H * dk, bias=False)
        self.o_proj = nn.Linear(H * dk, d, bias=False)
        self.head_gate = nn.Linear(d, H) if config.lin_head_gate else None
        self.gain = nn.Parameter(torch.ones(H, dk))
        self.g = nn.Parameter(torch.full((d,), config.lin_gate_init))
        # Fixed decay half-lives, log-uniform over lin_half_life_range and staggered by block so the blocks
        # together cover the range more densely. Plain floats, not a buffer: a buffer would be rounded by
        # model.half() and left uninitialized by from_pretrained's meta-device construction.
        self._half_life = branch_half_lives(config, block)
        self._half_life_on = {}  # device -> tensor: building it from the list is a blocking host-to-device copy
        self.stats = None  # set when a caller turns on `record_stats`
        self.record_stats = False

    @property
    def half_life(self):
        """[H] fp32, in tokens. Not a parameter and not saved."""
        return self._half_life_at(self.g.device)

    def _half_life_at(self, device):
        if device not in self._half_life_on:
            self._half_life_on[device] = torch.tensor(self._half_life, dtype=torch.float32, device=device)
        return self._half_life_on[device]

    def forward_on(self, device, x0, doc_pos, token_mask):
        """forward() computed on another GPU, for training and eval (no cache). The input and every
        parameter are copied there first, differentiably, so the gradients land on the parameters where
        they live. The result stays on `device`: move it back with to_device only when it is needed, and
        the branch runs alongside whatever the caller queues on its own GPU in between."""
        weights = {n: to_device(p, device, host_wait_back=True) for n, p in self.named_parameters()}
        x0 = to_device(x0, device, host_wait_back=True)
        doc_pos = to_device(doc_pos, device)
        token_mask = None if token_mask is None else to_device(token_mask, device)
        return self(x0, None, doc_pos, token_mask, weights=weights)

    def _sub(self, name, weights, *args):
        """Submodule `name` applied to args, with its parameters taken from `weights` when given."""
        module = getattr(self, name)
        if weights is None:
            return module(*args)
        return functional_call(module, {k: weights[f"{name}.{k}"] for k, _ in module.named_parameters()}, args)

    def forward(self, x0, cache: Cache | None, doc_pos, token_mask, weights=None):
        """`weights`: parameter name -> tensor to use in its place (forward_on's copies)."""
        B, T, _ = x0.shape
        H, dk = self.heads, self.head_dim
        gain, g = (self.gain, self.g) if weights is None else (weights["gain"], weights["g"])
        u = self._sub("conv", weights, self._sub("norm", weights, x0), cache, token_mask)
        q, k, v = (self._sub(proj, weights, u).view(B, T, H, dk) for proj in ("q_proj", "k_proj", "v_proj"))
        S_init = None
        if cache is not None:
            layer = cache.layers[self.layer_idx]
            if layer.is_recurrent_states_initialized[0]:
                S_init = layer.recurrent_states[0]
        with torch.autocast(device_type=x0.device.type, enabled=False):
            q, k, v = F.normalize(q.float(), dim=-1, eps=1e-6), F.normalize(k.float(), dim=-1, eps=1e-6), v.float()
            o, S = retention_chunked(q, k, v, math.log(2) / self._half_life_at(x0.device), doc_pos, S_init, self.chunk)
            y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps) * gain.float()
        if self.head_gate is not None:
            y = y * torch.sigmoid(self._sub("head_gate", weights, u).float())[..., None]  # [B, T, H, 1]; per token, so nothing to cache
        if cache is not None:
            cache.update_recurrent_state(S, self.layer_idx, state_idx=0)
        out = g.float() * self._sub("o_proj", weights, y.reshape(B, T, H * dk).to(self.o_proj.weight.dtype)).float()
        if self.record_stats:
            self.stats = {"pre_norm_max": o.detach().abs().amax(dim=(1, 2, 3)), "state_max": S.detach().abs().amax(dim=(1, 2, 3)),
                          "out_rms": out.detach().pow(2).mean(dim=(1, 2)).sqrt()}
        return out
