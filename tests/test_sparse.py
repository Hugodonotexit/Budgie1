"""CPU tests of the compressed + selected global attention (config.sparse_global, sparse.py): the vectorized code
against a token-by-token fp64 reference written from the rules in sparse.py's docstring, causality, a state carried
across calls, padding, gradients, the chunked long-sequence path, cached decoding and configuration loading."""

import json

import pytest
import torch
from torch.nn import functional as F

import budgie.sparse as sparse
from budgie import BudgieConfig, BudgieForCausalLM
from budgie.sparse import compress_kv, initial_state, sparse_attention
from test_model import tiny_model, tokens

SPEC = {"every": 1, "compress": 4, "select": 3, "group": 3, "local": 5}


def inputs(B=2, H=4, Hkv=2, T=53, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, H, T, D, generator=g, dtype=torch.float64), dim=-1) * 2
    k = F.normalize(torch.randn(B, Hkv, T, D, generator=g, dtype=torch.float64), dim=-1) * 2
    v = torch.randn(B, Hkv, T, D, generator=g, dtype=torch.float64)
    u = torch.randn(Hkv, D, generator=g, dtype=torch.float64)
    sinks = torch.randn(H, generator=g, dtype=torch.float64)
    return q, k, v, u, sinks


def naive_pool(k, v, u, m, key_ok=None):
    B, Hkv, T, D = k.shape
    nb = T // m
    kc, vc = torch.zeros(B, Hkv, nb, D, dtype=k.dtype), torch.zeros(B, Hkv, nb, D, dtype=k.dtype)
    for b in range(B):
        for h in range(Hkv):
            for j in range(nb):
                idx = [i for i in range(j * m, j * m + m) if key_ok is None or key_ok[b, i]]
                if not idx:
                    continue
                w = torch.softmax(torch.stack([k[b, h, i] @ u[h] for i in idx]), 0)
                pk = sum(wi * k[b, h, i] for wi, i in zip(w, idx))
                rms = lambda x: x.pow(2).mean().add(1e-6).sqrt()
                target = sum(wi * rms(k[b, h, i]) for wi, i in zip(w, idx))
                kc[b, h, j] = pk * target / rms(pk)
                vc[b, h, j] = sum(wi * v[b, h, i] for wi, i in zip(w, idx))
    return kc, vc


def reference(q, k, v, u, sinks, spec, scale, key_ok=None, q_ok=None):
    """One query at a time, straight from the rules in sparse.py's docstring."""
    m, n, g, W = spec["compress"], spec["select"], spec["group"], spec["local"]
    B, H, T, D = q.shape
    rep = H // k.shape[1]
    kc, vc = naive_pool(k, v, u, m, key_ok)
    nb = T // m
    real = lambda b, i: key_ok is None or bool(key_ok[b, i])
    blk = lambda b, j: any(real(b, i) for i in range(j * m, j * m + m))
    S = torch.zeros(B, T // g + 1, max(nb, 1), dtype=torch.float64)
    for b in range(B):
        for t in range(T):
            if q_ok is not None and not q_ok[b, t]:
                continue
            vis = [j for j in range((t // g * g) // m) if blk(b, j)]
            for h in range(H):
                if vis:
                    p = torch.softmax(torch.stack([q[b, h, t] @ kc[b, h // rep, j] * scale for j in vis]), 0)
                    for pj, j in zip(p, vis):
                        S[b, t // g, j] += pj
    out = torch.zeros_like(q)
    for b in range(B):
        for t in range(T):
            G, gs = t // g, (t // g) * g
            cand = [j for j in range(max(0, gs - W) // m) if blk(b, j)]
            chosen = []
            if cand and min(n, nb):
                sc = torch.full((nb,), float("-inf"), dtype=torch.float64)
                for j in cand:
                    sc[j] = S[b, G - 1, j] if G else 0.0
                top, idx = sc.topk(min(n, nb))
                chosen = [int(j) for j, s in zip(idx, top) if s > float("-inf")]
            for h in range(H):
                hk = h // rep
                logits, vals = [sinks[h]], [torch.zeros(D, dtype=torch.float64)]
                for j in range(gs // m):
                    if blk(b, j):
                        logits.append(q[b, h, t] @ kc[b, hk, j] * scale); vals.append(vc[b, hk, j])
                for j in chosen:
                    for i in range(j * m, j * m + m):
                        if real(b, i):
                            logits.append(q[b, h, t] @ k[b, hk, i] * scale); vals.append(v[b, hk, i])
                for i in range(max(0, gs - W), t + 1):
                    if real(b, i):
                        logits.append(q[b, h, t] @ k[b, hk, i] * scale); vals.append(v[b, hk, i])
                p = torch.softmax(torch.stack(logits), 0)
                out[b, h, t] = sum(pi * vi for pi, vi in zip(p, vals))
    return out


def run(q, k, v, u, sinks, spec, scale, key_ok=None, q_ok=None):
    kc, vc, blk_ok = compress_kv(k, v, u, spec["compress"], key_ok)
    return sparse_attention(q, k, v, kc, vc, blk_ok, p0=0, spec=spec, sinks=sinks, scale=scale, key_ok=key_ok, q_ok=q_ok)


def test_pooling_matches_naive():
    _, k, v, u, _ = inputs()
    key_ok = torch.ones(2, 53, dtype=torch.bool)
    key_ok[0, :9] = False                                                    # left padding: block 0 empty, block 2 partial
    for ok in (None, key_ok):
        kc, vc, blk_ok = compress_kv(k, v, u, 4, ok)
        nk, nv = naive_pool(k, v, u, 4, ok)
        assert (kc - nk).abs().max() <= 1e-12 and (vc - nv).abs().max() <= 1e-12
    assert blk_ok[0].tolist()[:3] == [False, False, True] and blk_ok[1].all()


@pytest.mark.parametrize("spec", [SPEC, {"every": 1, "compress": 5, "select": 2, "group": 4, "local": 4},
                                  {"every": 1, "compress": 3, "select": 50, "group": 1, "local": 2}])
def test_matches_reference(spec, monkeypatch):
    q, k, v, u, sinks = inputs()
    want = reference(q, k, v, u, sinks, spec, 0.35)
    got, _ = run(q, k, v, u, sinks, spec, 0.35)
    assert (got - want).abs().max() <= 1e-12
    monkeypatch.setattr(sparse, "BUDGET", 1)                                 # one query group per batch
    got, _ = run(q, k, v, u, sinks, spec, 0.35)
    assert (got - want).abs().max() <= 1e-12
    got, _ = run(q, k, v, u, None, spec, 0.35)                               # no sink
    assert (got - reference(q, k, v, u, torch.full_like(sinks, -1e30), spec, 0.35)).abs().max() <= 1e-12


def test_padding_matches_reference():
    q, k, v, u, sinks = inputs()
    key_ok = torch.ones(2, 53, dtype=torch.bool)
    key_ok[0, :11] = False
    key_ok[1, :2] = False
    got, _ = run(q, k, v, u, sinks, SPEC, 0.35, key_ok, key_ok)
    want = reference(q, k, v, u, sinks, SPEC, 0.35, key_ok, key_ok)
    assert (got - want)[key_ok[:, None, :, None].expand_as(got)].abs().max() <= 1e-12


def test_causal():
    q, k, v, u, sinks = inputs()
    base, _ = run(q, k, v, u, sinks, SPEC, 0.35)
    for cut in (7, 24, 40):
        q2, k2, v2 = q.clone(), k.clone(), v.clone()
        for t in (q2, k2, v2):
            t[:, :, cut:] = torch.randn_like(t[:, :, cut:])
        got, _ = run(q2, k2, v2, u, sinks, SPEC, 0.35)
        assert (got[:, :, :cut] - base[:, :, :cut]).abs().max() <= 1e-12, cut


def test_state_across_calls_and_initial_state():
    q, k, v, u, sinks = inputs()
    full, _ = run(q, k, v, u, sinks, SPEC, 0.35)
    kc, vc, blk_ok = compress_kv(k, v, u, 4)
    for cuts in ([10], [9, 12, 13, 30], list(range(1, 53))):
        outs, state, a = [], None, 0
        for b in cuts + [53]:
            o, state = sparse_attention(q[:, :, a:b], k[:, :, :b], v[:, :, :b], kc, vc, blk_ok, p0=a, spec=SPEC, sinks=sinks, scale=0.35, state=state)
            outs.append(o)
            a = b
        assert (torch.cat(outs, 2) - full).abs().max() <= 1e-12, cuts
    for s in (3, 12, 27):                                                    # the chunked path: start from the group before
        st = initial_state(q[:, :, s - 3:s], s, kc, blk_ok, 0.35, SPEC)
        o, _ = sparse_attention(q[:, :, s:], k, v, kc, vc, blk_ok, p0=s, spec=SPEC, sinks=sinks, scale=0.35, state=st)
        assert (o - full[:, :, s:]).abs().max() <= 1e-12, s


def test_gradients(monkeypatch):
    q, k, v, u, sinks = (t[:1].clone() if t.dim() == 4 else t.clone() for t in inputs(H=2, Hkv=1, T=15, D=4))
    monkeypatch.setattr(sparse, "BUDGET", 300)                               # several checkpointed batches of groups
    spec = {"every": 1, "compress": 3, "select": 2, "group": 2, "local": 3}
    leaves = [t.requires_grad_(True) for t in (q, k, v, u, sinks)]

    def f(q, k, v, u, sinks):
        kc, vc, blk_ok = compress_kv(k, v, u, 3)
        return sparse_attention(q, k, v, kc, vc, blk_ok, p0=0, spec=spec, sinks=sinks, scale=0.35)[0]
    assert torch.autograd.gradcheck(f, leaves, atol=1e-7)


def test_model_decode_generate_and_config(tmp_path):
    sp = {"every": 2, "compress": 4, "select": 2, "group": 3, "local": 5}
    m = tiny_model(sparse_global=sp, num_blocks=2)
    size = len(m.config.layer_pattern)
    assert [i for i, l in enumerate(m.model.layers) if l.self_attn.sparse] == [2 * size - 1]   # block 2's global layer
    ids = tokens(2, 50, bos_at=(20,))
    with torch.no_grad():
        full = m(input_ids=ids, use_cache=False).logits
        out = m(input_ids=ids[:, :22], use_cache=True)
        cache, got = out.past_key_values, [out.logits]
        for t in range(22, 50):
            out = m(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            got.append(out.logits)
        assert (torch.cat(got, 1) - full).abs().max() <= 1e-5
        gen = m.generate(input_ids=ids[:, :10], max_new_tokens=8, do_sample=False, num_beams=2, pad_token_id=0)
        assert gen.shape == (2, 18)
    m.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["sparse_global"] == sp
    back = BudgieForCausalLM.from_pretrained(tmp_path)
    with torch.no_grad():
        assert torch.equal(back(input_ids=ids, use_cache=False).logits, full)
    d = m.config.to_dict()
    del d["sparse_global"]                                                   # a config.json from before sparse_global
    assert BudgieConfig.from_dict(d).sparse_global is None
    assert BudgieConfig().sparse_global == {"every": 2, "compress": 64, "select": 16, "group": 16, "local": 128}
    with pytest.raises(ValueError):
        BudgieConfig(sparse_global={**sp, "local": 2})


def test_chunked_long_sequence_path_matches_plain():
    """The long-sequence mode's chunked attention half (forward, and its staged backward) == the plain layer."""
    from budgie.chunked import AttentionHalf
    torch.manual_seed(0)
    sp = {"every": 1, "compress": 4, "select": 2, "group": 3, "local": 5}
    m = tiny_model(sparse_global=sp, num_blocks=1).double()
    i = next(i for i, l in enumerate(m.model.layers) if l.self_attn.sparse)
    layer = m.model.layers[i]
    assert m.config.kv_owner(i) is None                                     # one block: the global layer owns its K/V
    T = 60
    x = torch.randn(2, T, 32, dtype=torch.float64)
    cos_sin = tuple(c.double() for c in m.model.rotary(torch.arange(T)[None]))
    xr = x.clone().requires_grad_(True)
    a, kv = layer.self_attn(layer.attn_norm(xr), cos_sin, None, None, None, None, 0)
    plain = xr + a
    g = torch.randn_like(plain)
    plain.backward(g)
    want = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
    layer.zero_grad()
    half = AttentionHalf(layer, cos_sin, 12)
    out = half.forward(x)
    assert (out[0] - plain).abs().max() <= 1e-12
    dx, _ = half.backward(x, (), (g, None, None, None))
    assert (dx - xr.grad).abs().max() <= 1e-7      # a dense global layer shows the same ~1e-8 here; parameter grads are exact
    for n, p in layer.named_parameters():
        if n in want:
            assert (p.grad - want[n]).abs().max() <= 1e-12 * max(1.0, want[n].abs().max().item()), n
    assert "self_attn.pool_score" in want
