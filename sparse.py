"""Compressed + selected global attention (config.sparse_global), after NSA (Yuan et al. 2025) and DeepSeek's CSA.

A query at absolute position t, in query group G = t // group (positions G*group .. G*group + group - 1), attends in
ONE softmax (with the layer's learned sink) over

    compressed   every block j of `compress` (m) keys that ends before the group starts (j < G*group // m), pooled
                 into one key and one value: a softmax over the block's keys of (k . pool_score[h]) weights both
                 (pool_score starts at 0 = the mean), and the pooled key is rescaled to the weighted mean RMS of
                 the keys it pools, so it scores on the same scale as a single key;
    selected     the `select` blocks with the highest score S_{G-1}[j], every key of them exactly, where S_{G-1} is
                 the softmax over the compressed blocks summed over all heads and over the queries of the PREVIOUS
                 group. Only blocks that end before the local window are candidates (j < (G*group - local) // m),
                 and local >= group, so the previous group saw every candidate;
    local        the keys G*group - local .. t exactly.

The three sets do not overlap except that a selected block is also in the compressed set. Selection is decided by
earlier queries only, so the layer is causal and a cached decode reproduces a full forward exactly. The score
needs no extra parameters or loss: the compressed attention is trained by the language-model loss, and blocks it
attends to are what gets selected (NSA's argument). top-k itself has no gradient.

Cost per query: (t / m) + select * m + local + group keys, so ~16k + 1k + 144 at 1M with the defaults (m 64,
select 16, local 128, group 16) instead of ~512k for a dense global layer. Attention scores are materialized, a
bounded number of query groups at a time (BUDGET), so this needs no custom kernel; with grad enabled each batch of
groups is checkpointed when there is more than one.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

BUDGET = 1 << 25   # attention-score (and gathered K/V) elements per batch of query groups
NEG = float("-inf")


def compress_kv(k, v, pool_score, m, key_ok=None, eps=1e-6):
    """k, v [B, kv_heads, Tk, D] -> pooled keys and values [B, kv_heads, Tk // m, D] in the compute dtype of q's
    softmax (fp32, or fp64 for fp64 inputs), and blk_ok [B, Tk // m] (None without padding): False for a block
    with no real key. pool_score [kv_heads, D]; key_ok [B, Tk] marks real keys. Blocks are independent, so a
    slice of whole blocks gives the same blocks as the whole sequence."""
    B, Hkv, Tk, D = k.shape
    nb = Tk // m
    acc = torch.promote_types(k.dtype, torch.float32)
    kb = k[:, :, :nb * m].reshape(B, Hkv, nb, m, D).to(acc)
    vb = v[:, :, :nb * m].reshape(B, Hkv, nb, m, D).to(acc)
    logit = torch.einsum("bhnmd,hd->bhnm", kb, pool_score.to(acc))
    blk_ok = None
    if key_ok is not None:
        okb = key_ok[:, :nb * m].reshape(B, 1, nb, m)
        blk_ok = okb.any(-1)[:, 0]
        logit = logit.masked_fill(~okb, NEG).masked_fill(~okb.any(-1, keepdim=True), 0.0)
        w = torch.softmax(logit, -1) * okb
    else:
        w = torch.softmax(logit, -1)
    pk = torch.einsum("bhnm,bhnmd->bhnd", w, kb)
    target = (w * kb.pow(2).mean(-1).add(eps).sqrt()).sum(-1)              # weighted mean RMS of the pooled keys
    kc = pk * (target / pk.pow(2).mean(-1).add(eps).sqrt())[..., None]
    vc = torch.einsum("bhnm,bhnmd->bhnd", w, vb)
    return kc, vc, blk_ok


def _fit(S, nb):
    return S[..., :nb] if S.shape[-1] >= nb else F.pad(S, (0, nb - S.shape[-1]))


def _visible_blocks(pos, g, m):
    """Compressed blocks j < this are visible to a query at position pos (any int tensor)."""
    return torch.div(pos, g, rounding_mode="floor") * g // m


@torch.no_grad()
def group_sums(q, pos0, kc, blk_ok, scale, g, m, q_ok, G0, nG):
    """Per query group G0 .. G0 + nG - 1: the sum over its queries (at positions pos0 .. pos0 + T - 1; others of the
    group contribute elsewhere) and over all heads of each query's softmax over the compressed blocks it sees.
    q [B, H, T, D]; returns [B, nG, nb] in the softmax dtype."""
    B, H, T, D = q.shape
    Hkv, nb = kc.shape[1], kc.shape[2]
    out = kc.new_zeros(B, nG, nb, dtype=torch.promote_types(kc.dtype, torch.float32))
    if nb == 0 or T == 0:
        return out
    rep = H // Hkv
    kcq = kc.to(q.dtype)
    blocks = torch.arange(nb, device=q.device)
    step = max(1, BUDGET // max(1, B * H * nb))
    for s in range(0, T, step):
        e = min(T, s + step)
        pos = torch.arange(pos0 + s, pos0 + e, device=q.device)
        lg = torch.einsum("bhrtd,bhnd->bhrtn", q[:, :, s:e].reshape(B, Hkv, rep, e - s, D), kcq).to(out.dtype) * scale
        mask = (blocks[None, :] < _visible_blocks(pos, g, m)[:, None])[None, None, None]
        if blk_ok is not None:
            mask = mask & blk_ok[:, None, None, None, :]
        any_ = mask.any(-1, keepdim=True)
        p = torch.softmax(lg.masked_fill(~mask, NEG).masked_fill(~any_, 0.0), -1) * any_
        p = p.sum((1, 2))                                                   # [B, t, nb]
        if q_ok is not None:
            p = p * q_ok[:, s:e, None].to(p.dtype)
        out.index_add_(1, torch.div(pos, g, rounding_mode="floor") - G0, p)
    return out


def _gather(x, tok):
    """x [B, Hkv, Tk, D], tok [B, nG, L] -> [B, Hkv, nG, L, D]."""
    bidx = torch.arange(x.shape[0], device=x.device)[:, None, None]
    return x[bidx, :, tok].permute(0, 3, 1, 2, 4)


def _attend_groups(qg, k, v, kc, vc, c_mask, tok, tok_ok, locc, loc_ok, sinks, scale):
    """One batch of query groups. qg [B, Hkv, rep, nG, g, D]; c_mask [B|1, nG, nb]; tok, tok_ok [B, nG, L] or None;
    locc [nG, Lw] (clamped local key positions); loc_ok [B|1, nG, g, Lw]. Returns [B, Hkv, rep, nG, g, D]."""
    B, Hkv, rep, nG, g, D = qg.shape
    cd = qg.dtype
    acc = torch.promote_types(cd, torch.float32)
    logits, values = [], []
    if sinks is not None:
        logits.append(sinks.to(acc).view(1, Hkv, rep, 1, 1, 1).expand(B, Hkv, rep, nG, g, 1))
        values.append(None)
    if kc.shape[2]:
        lc = torch.einsum("bhrngd,bhjd->bhrngj", qg, kc.to(cd)).to(acc) * scale
        logits.append(lc.masked_fill(~c_mask[:, None, None, :, None, :], NEG))
        values.append(("c", vc.to(cd)))
    if tok is not None:
        ks, vs = _gather(k, tok), _gather(v, tok)
        ls = torch.einsum("bhrngd,bhnld->bhrngl", qg, ks.to(cd)).to(acc) * scale
        logits.append(ls.masked_fill(~tok_ok[:, None, None, :, None, :], NEG))
        values.append(("g", vs.to(cd)))
    kl, vl = k[:, :, locc], v[:, :, locc]                                   # [B, Hkv, nG, Lw, D]
    ll = torch.einsum("bhrngd,bhnld->bhrngl", qg, kl.to(cd)).to(acc) * scale
    logits.append(ll.masked_fill(~loc_ok[:, None, None], NEG))
    values.append(("g", vl.to(cd)))
    lg = torch.cat(logits, -1)
    any_ = (lg > NEG).any(-1, keepdim=True)
    p = torch.softmax(lg.masked_fill(~any_, 0.0), -1) * any_
    out, at = 0, 0
    for part, val in zip(logits, values):
        w = part.shape[-1]
        pp = p[..., at:at + w].to(cd)
        at += w
        if val is None:
            continue                                                        # the sink's value is zero
        kind, vv = val
        out = out + (torch.einsum("bhrngj,bhjd->bhrngd", pp, vv) if kind == "c" else torch.einsum("bhrngl,bhnld->bhrngd", pp, vv))
    return out


def sparse_attention(q, k, v, kc, vc, blk_ok, *, p0, spec, sinks, scale, key_ok=None, q_ok=None, state=None):
    """q [B, H, T, D]: queries at absolute positions p0 .. p0 + T - 1. k, v [B, kv_heads, p0 + T, D]: the keys of
    positions 0 .. p0 + T - 1 (not expanded to H heads). kc, vc, blk_ok: compress_kv of them (whole blocks; extra
    blocks past p0 + T are ignored). key_ok [B, p0 + T] / q_ok [B, T]: real (non-pad) keys / queries, or None.
    state: what the queries before p0 left for selection (from a previous call), required when p0 is not 0:
    (group, S of the group before it, partial S of that group). Returns (output [B, H, T, D], state after the call)."""
    m, n, g, W = spec["compress"], spec["select"], spec["group"], spec["local"]
    B, H, T, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]
    if Tk != p0 + T:
        raise ValueError(f"sparse attention needs the keys of positions 0 .. {p0 + T - 1}, got {Tk}")
    rep = H // Hkv
    nb = min(kc.shape[2], Tk // m)
    kc, vc = kc[:, :, :nb], vc[:, :, :nb]
    blk_ok = None if blk_ok is None else blk_ok[:, :nb]
    dev = q.device
    G0, G1 = p0 // g, (p0 + T - 1) // g
    nG = G1 - G0 + 1

    # selection scores: S_{G-1} for every group G0 .. G1
    C = group_sums(q, p0, kc, blk_ok, scale, g, m, q_ok, G0, nG)
    if state is None:
        if p0:
            raise ValueError("sparse attention after position 0 needs the selection state of the earlier tokens")
        prev, part = C.new_zeros(B, nb), C.new_zeros(B, nb)
    else:
        grp, prev, part = state
        prev, part = _fit(prev.to(C.dtype), nb), _fit(part.to(C.dtype), nb)
        if grp == G0 - 1 and p0 % g == 0:
            prev, part = part, torch.zeros_like(part)
        elif grp != G0:
            raise ValueError(f"sparse attention state is for query group {grp}, but these queries start in group {G0}")
    C[:, 0] += part
    S_before = torch.cat([prev[:, None], C[:, :-1]], 1)                    # [B, nG, nb]
    new_state = (G1, S_before[:, -1].clone(), C[:, -1].clone())

    gs = torch.arange(G0, G1 + 1, device=dev) * g                          # each group's first position
    blocks = torch.arange(nb, device=dev)
    c_mask = (blocks[None, :] < (gs // m)[:, None])[None]                  # [1, nG, nb]
    if blk_ok is not None:
        c_mask = c_mask & blk_ok[:, None]
    nsel = min(n, nb)
    tok = tok_ok = None
    if nsel:
        cand = (blocks[None, :] < ((gs - W).clamp_min(0) // m)[:, None])[None]
        if blk_ok is not None:
            cand = cand & blk_ok[:, None]
        top, idx = S_before.masked_fill(~cand, NEG).topk(nsel, -1)        # [B, nG, nsel]
        tok = (idx[..., None] * m + torch.arange(m, device=dev)).reshape(B, nG, nsel * m)
        tok_ok = (top > NEG)[..., None].expand(B, nG, nsel, m).reshape(B, nG, nsel * m)
        if key_ok is not None:
            tok_ok = tok_ok & key_ok.gather(1, tok.reshape(B, -1)).reshape(B, nG, -1)
    loc = gs[:, None] - W + torch.arange(W + g, device=dev)[None]          # [nG, Lw]
    qpos = gs[:, None] + torch.arange(g, device=dev)[None]                 # [nG, g]
    loc_ok = ((loc >= 0) & (loc < Tk))[:, None, :] & (loc[:, None, :] <= qpos[:, :, None])   # [nG, g, Lw]
    locc = loc.clamp(0, Tk - 1)
    loc_ok = loc_ok[None]
    if key_ok is not None:
        loc_ok = loc_ok & key_ok[:, locc][:, :, None, :]

    front, back = p0 - G0 * g, (G1 + 1) * g - (p0 + T)
    qg = F.pad(q, (0, 0, front, back)).reshape(B, Hkv, rep, nG, g, D)
    L = tok.shape[-1] if tok is not None else 0
    per_group = B * H * g * (1 + nb + L + W + g) + 2 * B * Hkv * (L + W + g) * D
    step = max(1, BUDGET // per_group)
    outs = []
    for a in range(0, nG, step):
        z = min(nG, a + step)
        args = (qg[:, :, :, a:z], k, v, kc, vc, c_mask[:, a:z], None if tok is None else tok[:, a:z],
                None if tok is None else tok_ok[:, a:z], locc[a:z], loc_ok[:, a:z], sinks, scale)
        if torch.is_grad_enabled() and nG > step:
            outs.append(checkpoint(_attend_groups, *args, use_reentrant=False))
        else:
            outs.append(_attend_groups(*args))
    out = torch.cat(outs, 3).reshape(B, H, nG * g, D)[:, :, front:front + T]
    return out.to(q.dtype), new_state


def initial_state(q_before, p0, kc, blk_ok, scale, spec, q_ok=None):
    """The state for queries starting at p0 (a multiple of `group`), from the queries of the group just before it,
    q_before [B, H, group, D] at positions p0 - group .. p0 - 1. None when p0 is 0."""
    if p0 == 0:
        return None
    g, m = spec["group"], spec["compress"]
    if p0 % g or q_before.shape[2] != g:
        raise ValueError(f"initial_state needs p0 a multiple of {g} and the {g} queries before it")
    nb = min(kc.shape[2], p0 // m)
    S = group_sums(q_before, p0 - g, kc[:, :, :nb], None if blk_ok is None else blk_ok[:, :nb], scale, g, m, q_ok, p0 // g - 1, 1)
    return p0 // g - 1, torch.zeros_like(S[:, 0]), S[:, 0]
