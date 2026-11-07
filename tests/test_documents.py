"""Document masking (documents.py): packed documents give exactly what each document gives alone."""

import pytest
import torch

from budgie import patterns as patterns_module
from budgie.offload import HostPool
from budgie.patterns import AttentionPattern
from budgie.sdpa import sdpa

from test_model import BOS, VOCAB, tiny_model

# The retention branch computes in fp32 whatever the model's dtype, and its chunks fall differently inside a packed row
# than in a document alone, so with it the comparison is to fp32 rounding; everything else is compared in fp64.
FP32 = dict(rtol=1e-4, atol=2e-6)
FP64 = dict(rtol=1e-9, atol=1e-9)


def close(a, b, tol=2e-5):
    """Gradients: the largest difference against the largest entry (the loss head and the retention branch are fp32: ~1e-6 apart)."""
    assert (a - b).abs().max() <= tol * b.abs().max().clamp_min(1e-12)

# documents of lengths that are not multiples of any dilation, the sparse layer's block (4) or its group (3)
ROW0 = [13, 29, 7, 31]          # 80 tokens
ROW1 = [80]                     # one document: an ordinary row


def packed(rows, seed=3):
    g = torch.Generator().manual_seed(seed)
    T = sum(rows[0])
    ids = torch.randint(3, VOCAB, (len(rows), T), generator=g)
    starts = torch.zeros(len(rows), T, dtype=torch.bool)
    for b, lens in enumerate(rows):
        at = 0
        for n in lens:
            starts[b, at] = True
            ids[b, at] = BOS
            at += n
    return ids, starts


def pieces(rows):
    return [(b, sum(lens[:i]), sum(lens[:i + 1])) for b, lens in enumerate(rows) for i in range(len(lens))]


@pytest.mark.parametrize("branch", [True, False])
def test_each_document_comes_out_as_if_alone(branch):
    model = tiny_model(branch).double()
    ids, starts = packed([ROW0, ROW1])
    with torch.no_grad():
        got = model.model(ids, document_starts=starts, use_cache=False).last_hidden_state
        for b, s, e in pieces([ROW0, ROW1]):
            alone = model.model(ids[b:b + 1, s:e], use_cache=False).last_hidden_state[0]
            torch.testing.assert_close(got[b, s:e], alone, **(FP32 if branch else FP64))


def test_without_masking_the_documents_do_see_each_other():
    model = tiny_model().double()
    ids, starts = packed([ROW0])
    with torch.no_grad():
        plain = model.model(ids, use_cache=False).last_hidden_state
        masked = model.model(ids, document_starts=starts, use_cache=False).last_hidden_state
    assert (plain[0, 13:] - masked[0, 13:]).abs().max() > 1e-3     # the 2nd document on reads the 1st without it
    torch.testing.assert_close(plain[0, :13], masked[0, :13], rtol=0, atol=1e-12)


def test_no_start_after_position_0_is_an_ordinary_forward():
    model = tiny_model()
    ids, _ = packed([ROW1, ROW1])
    starts = torch.zeros_like(ids, dtype=torch.bool)
    starts[:, 0] = True
    with torch.no_grad():
        a = model(ids, use_cache=False).logits
        b = model(ids, document_starts=starts, use_cache=False).logits
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def _loss_sum_and_grads(model, ids, labels, **kw):
    model.zero_grad(set_to_none=True)
    out = model(ids, labels=labels, use_cache=False, **kw)
    n = int((labels[:, 1:] >= 0).sum())
    (out.loss * n).backward()
    return (out.loss * n).item(), {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}


def test_gradients_equal_the_sum_over_documents_alone():
    model = tiny_model().double().train()
    ids, starts = packed([ROW0, ROW1])
    labels = ids.masked_fill(starts, -100)          # a document's first token is not predicted from the one before
    total, grads = _loss_sum_and_grads(model, ids, labels, document_starts=starts)
    ref_total, ref = 0.0, {}
    for b, s, e in pieces([ROW0, ROW1]):
        t, g = _loss_sum_and_grads(model, ids[b:b + 1, s:e], labels[b:b + 1, s:e])
        ref_total += t
        for k, v in g.items():
            ref[k] = ref.get(k, 0) + v
    assert total == pytest.approx(ref_total, rel=1e-6)
    assert set(grads) == set(ref)
    for k in grads:
        close(grads[k], ref[k])


def test_long_context_mode_with_documents_matches_the_ordinary_path():
    """Offloaded halves, chunked (chunk 12 < most documents; a multiple of the sparse layer's query group, 3) and the
    segmented retention branch (segment 16, a multiple of its chunk)."""
    model = tiny_model().double().train()
    ids, starts = packed([ROW0, ROW1])
    labels = ids.masked_fill(starts, -100)
    total, ref = _loss_sum_and_grads(model, ids, labels, document_starts=starts)
    model.model.set_long_context_mode(HostPool(), branch_segment=16, chunk=12)
    try:
        got_total, got = _loss_sum_and_grads(model, ids, labels, document_starts=starts)
        with torch.no_grad():                   # the no-grad chunked forward of evaluation
            ev = model.model(ids, document_starts=starts, use_cache=False).last_hidden_state
    finally:
        model.model.set_long_context_mode(None)
    with torch.no_grad():
        ref_ev = model.model(ids, document_starts=starts, use_cache=False).last_hidden_state
    assert got_total == pytest.approx(total, rel=1e-6)
    assert set(got) == set(ref)
    for k in ref:
        close(got[k], ref[k])
    torch.testing.assert_close(ev, ref_ev, **FP32)


def test_document_masking_refuses_a_cache_and_padding():
    model = tiny_model()
    ids, starts = packed([ROW0])
    with pytest.raises(ValueError):
        model(ids, document_starts=starts, use_cache=True)
    mask = torch.ones_like(ids)
    mask[:, -3:] = 0
    with pytest.raises(ValueError):
        model(ids, attention_mask=mask, document_starts=starts, use_cache=False)


def _window_reference(q, k, v, scale, sinks, window, tail=False):
    """sdpa_window's semantics on the CPU: causal (aligned bottom right when tail), keys within `window`."""
    Tq, Tk = q.shape[2], k.shape[2]
    qp = torch.arange(Tk - Tq, Tk) if tail else torch.arange(Tq)
    rel = qp[:, None] - torch.arange(Tk)[None]
    ok = rel >= 0
    if window is not None:
        ok = ok & (rel < window)
    return sdpa(q, k, v, scale, sinks, mask=ok[None, None])


@pytest.mark.parametrize("window,dilation", [(5, 3), (None, 4), (4, 1)])
def test_a_chunk_ending_off_the_dilation_is_padded(monkeypatch, window, dilation):
    """attend_chunk's kernel path (CUDA only, so stood in for by a CPU reference of the kernel) for the last chunk of a
    document whose length is not a multiple of the dilation."""
    monkeypatch.setattr(patterns_module, "window_attention_supported", lambda q, sinks: True)
    monkeypatch.setattr(patterns_module, "sdpa_window", _window_reference)
    p = AttentionPattern(window, dilation)
    g = torch.Generator().manual_seed(0)
    B, H, Hkv, T, D = 2, 4, 2, 31, 8
    q = torch.randn(B, H, T, D, generator=g, dtype=torch.float64)
    k, v = (torch.randn(B, Hkv, T, D, generator=g, dtype=torch.float64) for _ in range(2))
    sinks = torch.randn(H, generator=g, dtype=torch.float64)
    rep = lambda t: t.repeat_interleave(H // Hkv, 1)
    ref = p.attend_masked(q, rep(k), rep(v), torch.arange(T), torch.arange(T), None, sinks, D ** -0.5)
    start = 24 // dilation * dilation
    got = p.attend_chunk(q[:, :, start:], k, v, start, sinks, D ** -0.5, H // Hkv)
    torch.testing.assert_close(got, ref[:, :, start:], rtol=1e-10, atol=1e-12)
