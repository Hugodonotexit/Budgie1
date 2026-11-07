"""A linear-attention branch that runs in parallel with a block's local layers and writes one gated residual
term just before the block's global layer. Two state updates (config.lin_rule):

"gated_delta" (default for new models): Gated DeltaNet, see gated_delta_reference below; per-token decay and
write strength, and the delta rule replaces the value stored at a key instead of adding to it.

"retention" (configs written before lin_rule): fixed-decay RetNet-style retention.
Per head, with decay lambda = ln2 / half_life and gamma = exp(-lambda) (fixed, not learned):

    S_t = gamma * S_{t-1} + (1 - gamma) * k_t v_t^T           fp32, zeroed at a document start
    o_t = q_t^T S_t / (1 - gamma^{n_t})                       n_t = tokens since the document start, t included

Dividing by (1 - gamma^n) turns S into a weighted average, so its scale stays bounded at any position
(unnormalized it would reach about 1e5 * |v| at a 64k half-life, past fp16's range). q and k are L2-
normalized. `retention_reference` is the token-by-token fp64 oracle; `retention_chunked` computes the same
thing `chunk` tokens at a time with every exponent <= 0.
"""

import functools
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
from .offload import offload_checkpoint


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
# Gated DeltaNet (lin_rule = "gated_delta")
# ---------------------------------------------------------------------------
#
# Per head, with a per-token decay alpha_t = exp(-lam_t) and write strength beta_t in (0, 1):
#
#     S_t = alpha_t S_{t-1} + beta_t k_t (v_t - alpha_t S_{t-1}^T k_t)^T  =  alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
#     o_t = q_t^T S_t
#
# S [dk, dv] maps a key to a value. Writing v_t at k_t first removes what S already returns for k_t (the delta
# rule), so a new value replaces the old one at that key instead of adding to it. With |k| = 1 and beta <= 1 every
# step is a contraction plus a bounded write, so S stays bounded with no normalization.


def gated_delta_reference(q, k, v, lam, beta, doc_pos, S_init=None):
    """Token-by-token fp64 oracle. q, k: [B, T, H, dk]; v: [B, T, H, dv]; lam, beta: [B, T, H]; doc_pos: [B, T].
    Returns (o [B, T, H, dv], S [B, H, dk, dv]). S is zeroed at every token with doc_pos == 1."""
    q, k, v, lam, beta = (t.double() for t in (q, k, v, lam, beta))
    B, T, H, dk = q.shape
    S = torch.zeros(B, H, dk, v.shape[-1], dtype=torch.float64, device=q.device) if S_init is None else S_init.double().clone()
    out = []
    for t in range(T):
        S = torch.where((doc_pos[:, t] == 1)[:, None, None, None], torch.zeros_like(S), S)
        S = torch.exp(-lam[:, t])[..., None, None] * S
        kt = k[:, t]
        err = v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, S)                             # what to write minus what is there
        S = S + beta[:, t][..., None, None] * kt[..., :, None] * err[..., None, :]
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(out, 1), S


def _gated_delta_chunk_terms(q, k, v, dp, lam, beta):
    """Everything about a batch of chunks that does not depend on the state carried into it. q, k [B, n, L, H, dk],
    v [B, n, L, H, dv], dp [B, n, L], lam and beta [B, n, L, H]. With c_i the within-chunk cumulative log decay
    (Gamma_i = e^{c_i}) and r_ij = e^{c_i - c_j} for j <= i in the same document (0 otherwise), the per-token write
    u_i = beta_i (v_i - Gamma_i k_i^T S_prev - sum_{j<i} r_ij (k_j . k_i) u_j) is a unit lower-triangular system
    (I + A) U = beta V - beta Gamma K S_prev. Splitting it into U = Uv - W S_prev gives, all independent of S_prev:
        O_intra [B, n, H, L, dv]    P Uv, with P_ij = r_ij (q_i . k_j)
        Q_eff   [B, n, H, L, dk]    Gamma q - P W:  o = O_intra + Q_eff S_prev
        M       [B, n, H, dk, dk]   Gamma_L I - K~^T W  and
        N       [B, n, H, dk, dv]   K~^T Uv:  S_next = M S_prev + N, with K~_j = (Gamma_L / Gamma_j) k_j
    Every exponent is <= 0. Gamma (on S_prev) is 0 for tokens outside the document carried into the chunk, and K~
    covers only the chunk's last document, so a document start cuts the chain exactly as in the reference."""
    L, dev, dt = q.shape[2], q.device, q.dtype
    i = torch.arange(L, device=dev)
    lag = i[:, None] - i[None, :]
    same = (lag >= 0) & (lag < dp[..., :, None])                                         # j <= i and the same document, [B, n, L, L]
    c = (-lam).cumsum(2).transpose(2, 3)                                                  # [B, n, H, L]
    r = (c[..., :, None] - c[..., None, :]).masked_fill(~same[:, :, None], float("-inf")).exp()   # [B, n, H, L, L]
    qT, kT, vT = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)                  # [B, n, H, L, d]
    bT = beta.transpose(2, 3)                                                             # [B, n, H, L]
    A = (bT[..., :, None] * r * (kT @ kT.transpose(-1, -2))).tril(-1)                     # strictly below the diagonal
    P = r * (qT @ kT.transpose(-1, -2))
    cont = (dp > (i + 1))[:, :, None]                                                     # [B, n, 1, L]: in the carried document
    gam = c.exp() * cont                                                                  # [B, n, H, L]
    eye = torch.eye(L, device=dev, dtype=dt)
    rhs = torch.cat([bT[..., None] * vT, (bT * gam)[..., None] * kT], -1)
    X = torch.linalg.solve_triangular(A + eye, rhs, upper=False, unitriangular=True)
    Uv, W = X.split([vT.shape[-1], kT.shape[-1]], -1)
    O_intra = P @ Uv
    Q_eff = gam[..., None] * qT - P @ W
    in_last = ((L - 1 - i) < dp[..., L - 1:L])[:, :, None]                                # [B, n, 1, L]: in the chunk's last document
    Kt = ((c[..., L - 1:] - c).exp() * in_last)[..., None] * kT                           # [B, n, H, L, dk]
    a = (c[..., L - 1].exp() * cont[..., L - 1])[..., None, None]                         # [B, n, H, 1, 1]
    M = a * torch.eye(kT.shape[-1], device=dev, dtype=dt) - Kt.transpose(-1, -2) @ W
    N = Kt.transpose(-1, -2) @ Uv
    return O_intra, Q_eff, M, N


def _gated_delta_run_chunks(q, k, v, dp, lam, beta, S):
    """Chunk terms for all n chunks at once, then the only sequential part, S_c = M_c S_{c-1} + N_c.
    Returns (o [B, n*L, H, dv], S after the last chunk)."""
    O_intra, Q_eff, M, N = _gated_delta_chunk_terms(q, k, v, dp, lam, beta)
    prev = []
    for c in range(M.shape[1]):
        prev.append(S)
        S = torch.baddbmm(N[:, c].flatten(0, 1), M[:, c].flatten(0, 1), S.flatten(0, 1)).view_as(S)
    prev = torch.stack(prev, 1)                                                           # state entering each chunk, [B, n, H, dk, dv]
    O = O_intra + Q_eff @ prev                                                            # [B, n, H, L, dv]
    return O.transpose(2, 3).flatten(1, 2), S


def gated_delta_chunked(q, k, v, lam, beta, doc_pos, S_init=None, chunk=64):
    """The Gated DeltaNet recurrence `chunk` tokens at a time; same arguments and results as gated_delta_reference.
    Within a chunk it solves one L x L unit-triangular system per head (the WY form of the delta rule), so the
    cost and memory per chunk grow with chunk^2. As in retention_chunked, the `T % chunk` tokens left over form
    one shorter chunk of their own."""
    B, T, H, dk = q.shape
    dv, dt = v.shape[-1], q.dtype
    lam, beta = lam.to(dt), beta.to(dt)
    S = torch.zeros(B, H, dk, dv, dtype=dt, device=q.device) if S_init is None else S_init.to(dt)
    full = T // chunk * chunk
    outs = []
    for lo, hi, L in ((0, full, chunk), (full, T, T - full)):
        if hi > lo:
            n = (hi - lo) // L
            o, S = _gated_delta_run_chunks(q[:, lo:hi].reshape(B, n, L, H, dk), k[:, lo:hi].reshape(B, n, L, H, dk),
                                           v[:, lo:hi].reshape(B, n, L, H, dv), doc_pos[:, lo:hi].reshape(B, n, L),
                                           lam[:, lo:hi].reshape(B, n, L, H), beta[:, lo:hi].reshape(B, n, L, H), S)
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

    A widened branch (`config.lin_stages()` = [B, ..., H], see grow_branch.py): the B heads of the first
    stage stay exactly where the B-head grid put them, because their weights were trained for those decays.
    At each widening from P heads to N, head P + k, with j = k % P and m = 1 + k // P, goes m / N past
    existing head j (wrapping around the range), so the union is one even grid of N points spaced 1/N apart."""
    lo, hi = config.lin_half_life_range
    stages, off = config.lin_stages(), block / config.num_blocks
    base = stages[0]
    log_lo, span = math.log2(lo), math.log2(hi) - math.log2(lo)
    out = [2.0 ** (log_lo + span * (j + off) / base) for j in range(base)]
    frac = [(j + off) / base for j in range(base)]
    for prev, n in zip(stages, stages[1:]):
        for k in range(n - prev):
            j, m = k % prev, 1 + k // prev
            frac.append((frac[j] + m / n) % 1.0)
            out.append(2.0 ** (log_lo + span * frac[-1]))
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
        # Gated DeltaNet: per token and head, decay lam = lam_h * softplus(a_proj(u) + dt_bias) / ln2 (exactly the
        # head's half-life lam_h while a_proj = 0 and dt_bias = 0, as initialized) and write strength
        # beta = sigmoid(b_proj(u)). The half-life grid stays a fixed float per head, so the learned parameters
        # sit near 0, where fp16 resolves small updates.
        self.rule = config.lin_rule
        if self.rule == "gated_delta":
            self.a_proj = nn.Linear(d, H, bias=False)
            self.b_proj = nn.Linear(d, H)
            self.dt_bias = nn.Parameter(torch.zeros(H))
        self.gain = nn.Parameter(torch.ones(H, dk))
        self.g = nn.Parameter(torch.full((d,), config.lin_gate_init))
        # Fixed decay half-lives, log-uniform over lin_half_life_range and staggered by block so the blocks
        # together cover the range more densely. Plain floats, not a buffer: a buffer would be rounded by
        # model.half() and left uninitialized by from_pretrained's meta-device construction.
        self._half_life = branch_half_lives(config, block)
        self._half_life_on = {}  # device -> tensor: building it from the list is a blocking host-to-device copy
        self.stats = None  # set when a caller turns on `record_stats`
        self.record_stats = False
        self._head_mask = None      # see set_head_mask
        self._head_mask_on = {}

    def set_head_mask(self, mask):
        """Multiply every head's output by mask[h] (a list of `lin_heads` floats, or None for no mask) before o_proj.
        A zero silences a head completely: no output, and so no gradient for its q/k/v rows, o_proj columns,
        gate row or gain. For experiments (config.BRANCH_ZERO_HEADS); not saved with the model."""
        if mask is not None and len(mask) != self.heads:
            raise ValueError(f"head mask has {len(mask)} entries for {self.heads} heads")
        self._head_mask = None if mask is None else [float(m) for m in mask]
        self._head_mask_on = {}

    def _mask_at(self, device):
        if device not in self._head_mask_on:
            self._head_mask_on[device] = torch.tensor(self._head_mask, dtype=torch.float32, device=device)
        return self._head_mask_on[device]

    @property
    def half_life(self):
        """[H] fp32, in tokens. Not a parameter and not saved."""
        return self._half_life_at(self.g.device)

    def _half_life_at(self, device):
        if device not in self._half_life_on:
            self._half_life_on[device] = torch.tensor(self._half_life, dtype=torch.float32, device=device)
        return self._half_life_on[device]

    def forward_on(self, device, x0, doc_pos, token_mask, conv_pos=None):
        """forward() computed on another GPU, for training and eval (no cache). The input and every
        parameter are copied there first, differentiably, so the gradients land on the parameters where
        they live. The result stays on `device`: move it back with to_device only when it is needed, and
        the branch runs alongside whatever the caller queues on its own GPU in between."""
        weights = {n: to_device(p, device, host_wait_back=True) for n, p in self.named_parameters()}
        x0 = to_device(x0, device, host_wait_back=True)
        doc_pos = to_device(doc_pos, device)
        token_mask = None if token_mask is None else to_device(token_mask, device)
        conv_pos = None if conv_pos is None else to_device(conv_pos, device)
        return self(x0, None, doc_pos, token_mask, weights=weights, conv_pos=conv_pos)

    def _sub(self, name, weights, *args):
        """Submodule `name` applied to args, with its parameters taken from `weights` when given."""
        module = getattr(self, name)
        if weights is None:
            return module(*args)
        return functional_call(module, {k: weights[f"{name}.{k}"] for k, _ in module.named_parameters()}, args)

    def forward(self, x0, cache: Cache | None, doc_pos, token_mask, weights=None, S_init=None, drop=0, return_state=False,
                conv_pos=None):
        """`weights`: parameter name -> tensor to use in its place (forward_on's copies). `conv_pos` [B, T of x0]:
        document masking for the conv (documents.py); the state already restarts wherever doc_pos is 1.
        `S_init`, `drop`, `return_state` are for forward_segmented: the state carried in from the previous
        segment, the number of leading tokens of x0 that only supply the conv's context (their outputs are
        discarded, and `doc_pos` covers only the tokens after them), and whether to return (out, S)."""
        B, H, dk = x0.shape[0], self.heads, self.head_dim
        gain, g = (self.gain, self.g) if weights is None else (weights["gain"], weights["g"])
        u = self._sub("conv", weights, self._sub("norm", weights, x0), cache, token_mask, conv_pos)
        if drop:
            u = u[:, drop:]
        T = u.shape[1]
        q, k, v = (self._sub(proj, weights, u).view(B, T, H, dk) for proj in ("q_proj", "k_proj", "v_proj"))
        if cache is not None:
            layer = cache.layers[self.layer_idx]
            if layer.is_recurrent_states_initialized[0]:
                S_init = layer.recurrent_states[0]
        with torch.autocast(device_type=x0.device.type, enabled=False):
            q, k, v = F.normalize(q.float(), dim=-1, eps=1e-6), F.normalize(k.float(), dim=-1, eps=1e-6), v.float()
            lam_h = math.log(2) / self._half_life_at(x0.device)
            if self.rule == "gated_delta":
                dt_bias = self.dt_bias if weights is None else weights["dt_bias"]
                lam = lam_h * F.softplus(self._sub("a_proj", weights, u).float() + dt_bias.float()) / math.log(2)
                beta = torch.sigmoid(self._sub("b_proj", weights, u).float())
                o, S = gated_delta_chunked(q, k, v, lam, beta, doc_pos, S_init, self.chunk)
            else:
                o, S = retention_chunked(q, k, v, lam_h, doc_pos, S_init, self.chunk)
            y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps) * gain.float()
        if self.head_gate is not None:
            y = y * torch.sigmoid(self._sub("head_gate", weights, u).float())[..., None]  # [B, T, H, 1]; per token, so nothing to cache
        if self._head_mask is not None:
            y = y * self._mask_at(y.device)[:, None]                                       # [H, 1] against [B, T, H, dk]
        if cache is not None:
            cache.update_recurrent_state(S, self.layer_idx, state_idx=0)
        out = g.float() * self._sub("o_proj", weights, y.reshape(B, T, H * dk).to(self.o_proj.weight.dtype)).float()
        if self.record_stats:
            self.stats = {"pre_norm_max": o.detach().abs().amax(dim=(1, 2, 3)), "state_max": S.detach().abs().amax(dim=(1, 2, 3)),
                          "out_rms": out.detach().pow(2).mean(dim=(1, 2)).sqrt()}
        return (out, S) if return_state else out

    def _segment(self, x_ext, S_in, doc_pos, conv_pos=None, *, drop):
        out, S = self.forward(x_ext, None, doc_pos, None, S_init=S_in, drop=drop, return_state=True, conv_pos=conv_pos)
        return out.to(x_ext.dtype), S

    def forward_segmented(self, x0, doc_pos, segment, pool=None, conv_pos=None):
        """forward() without a cache, `segment` tokens at a time, for sequences whose whole-sequence activations
        do not fit (the branch needs ~100 MiB per 1k tokens in forward plus backward). The recurrent state is
        carried from one segment to the next, the conv gets its (width - 1) tokens of context by overlapping
        the segments, and with `pool` every segment is checkpointed with its input in pinned host memory, so
        backward only ever holds one segment. The result equals forward()'s: the segment is a multiple of
        the chunk, so the chunk boundaries are the same. Returns [B, T, d] in x0's dtype."""
        B, T, _ = x0.shape
        segment = max(self.chunk, segment // self.chunk * self.chunk)
        context = self.conv.width - 1
        outs, S = [], None
        for start in range(0, T, segment):
            end, lo = min(start + segment, T), max(0, start - context)
            run = functools.partial(self._segment, drop=start - lo)
            extra = () if conv_pos is None else (conv_pos[:, lo:end],)
            if pool is not None:
                out, S = offload_checkpoint(run, pool, x0[:, lo:end], S, doc_pos[:, start:end], *extra)
            else:
                out, S = run(x0[:, lo:end], S, doc_pos[:, start:end], *extra)
            outs.append(out)
        return torch.cat(outs, 1)
