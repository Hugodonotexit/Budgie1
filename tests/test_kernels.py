"""The optional fused kernels (budgie/kernels.py: Liger RMSNorm / RoPE / cross entropy, xformers attention) against the
plain PyTorch paths, on the real model code in fp16 (the training dtype) at the live model's proportions (head_dim 128,
adaptive softmax, fp32 residual stream). Each test runs the model with the kernels off, then on, and compares
loss, per-token NLL and every parameter's gradient; the fp32 model is the yardstick, so "no worse than PyTorch's own fp16
error" is what is asserted, not bitwise equality.

Needs a CUDA GPU and the packages (liger-kernel, xformers); skipped without them. From the repository root:
python -m pytest tests/test_kernels.py -s        (on a busy box pick a free GPU: CUDA_VISIBLE_DEVICES=2 ...)
"""

import copy
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from budgie import BudgieConfig, BudgieForCausalLM, kernels  # noqa: E402

if kernels.LIGER_ERROR or kernels.XFORMERS_ERROR:
    pytest.skip(f"kernel packages unusable: {kernels.LIGER_ERROR or kernels.XFORMERS_ERROR}", allow_module_level=True)

DEV = "cuda"
AT_IMPORT = dict(kernels.FLAGS)   # what the package does by itself, before any test configures anything
TYPES = {   # windows shorter than the 192-token test sequences, so the windowed paths really cut keys
    "S": {"window": 64, "dilation": 1, "conv": 3, "rope": True, "kv_heads": 2},
    "D4": {"window": 24, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 2},
    "D8": {"window": 32, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 2},
    "D16": {"window": 40, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 2},
    "G": {"window": None, "dilation": 1, "conv": 4, "rope": False, "kv_heads": 4},
}


@pytest.fixture(autouse=True)
def restore_flags():
    saved = dict(kernels.FLAGS)
    yield
    kernels.FLAGS.update(saved)


def make_model(dtype=torch.float16, seed=0, **kw):
    cfg = dict(vocab_size=6000, hidden_size=256, num_attention_heads=4, head_dim=128, attention_types=TYPES, num_blocks=1,
               intermediate_size=(512, 512, 512), adaptive_cutoffs=(256, 1024), ngram_buckets=4096, linear_branch="none",
               max_position_embeddings=1024)
    cfg.update(kw)
    torch.manual_seed(seed)
    m = BudgieForCausalLM(BudgieConfig(**cfg))
    with torch.no_grad():   # identity convs, zero sinks and a zero cluster table would hide wiring mistakes
        for n, p in m.named_parameters():
            if "conv" in n:
                p.add_(torch.randn_like(p) * 0.3)
            elif "sinks" in n:
                p.normal_(0, 1.0)
            elif "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
    return m.to(DEV, dtype)


def batch(B=2, T=192, vocab=6000, seed=1, ignore=True):
    g = torch.Generator().manual_seed(seed)
    # a skewed, Zipf-like draw so every adaptive cluster is hit, with some ignored labels
    ids = (torch.rand(B, T, generator=g) ** 3 * vocab).long().clamp_max(vocab - 1)
    labels = ids.clone()
    if ignore:
        labels[:, ::13] = -100
    return ids.to(DEV), labels.to(DEV)


def run(model, ids, labels, scale=1.0):
    """(loss, {parameter: gradient}) of one forward/backward at loss scale `scale`."""
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    (loss * scale).backward()
    return loss.detach().float(), {n: p.grad.detach().float() / scale for n, p in model.named_parameters() if p.grad is not None}


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def compare(name, kernel_run, torch_run, truth):
    """Gradient error of the kernel run against the fp32 truth must be no worse than PyTorch's own fp16 error."""
    worst = 0.0
    for n, g_truth in truth.items():
        e_torch, e_kernel = rel(torch_run[n], g_truth), rel(kernel_run[n], g_truth)
        assert torch.isfinite(kernel_run[n]).all(), f"{name}: non-finite gradient for {n}"
        assert e_kernel <= max(1.5 * e_torch, 0.02), f"{name}: {n} gradient error {e_kernel:.4f} vs PyTorch fp16's {e_torch:.4f}"
        worst = max(worst, e_kernel)
    return worst


@pytest.fixture(scope="module")
def reference():
    """fp16 and fp32 runs with the kernels off (the fp32 weights are the fp16 ones, so the two models are the same function)."""
    kernels.configure(liger=False, xformers=False)
    m16 = make_model()
    ids, labels = batch()
    m32 = copy.deepcopy(m16).float()
    loss16, g16 = run(m16, ids, labels)
    loss32, g32 = run(m32, ids, labels)
    return dict(model=m16, ids=ids, labels=labels, loss16=loss16, g16=g16, loss32=loss32, g32=g32)


@pytest.mark.parametrize("which", ["rms_norm", "rope", "loss", "embedding", "attention", "all"])
def test_training_step_matches_pytorch(reference, which):
    kernels.configure(liger=False, xformers=False)
    kernels.configure(**({"liger": True, "xformers": True} if which == "all" else {which: True}))
    loss, grads = run(reference["model"], reference["ids"], reference["labels"], scale=8192.0)   # the trainer's initial fp16 loss scale
    assert abs(loss.item() - reference["loss32"].item()) <= max(2 * abs(reference["loss16"].item() - reference["loss32"].item()), 2e-3), \
        f"{which}: loss {loss.item():.5f}, fp32 {reference['loss32'].item():.5f}, PyTorch fp16 {reference['loss16'].item():.5f}"
    worst = compare(which, grads, reference["g16"], reference["g32"])
    print(f"\n  {which:10s} loss {loss.item():.5f} (fp32 {reference['loss32'].item():.5f}, PyTorch fp16 {reference['loss16'].item():.5f}); worst gradient error vs fp32 {worst:.4f}")


def overflowing(model, ids, labels, scale):
    model.zero_grad(set_to_none=True)
    (model(input_ids=ids, labels=labels, use_cache=False).loss * scale).backward()
    return {n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()}


def test_kernels_add_no_fp16_overflow_at_a_high_loss_scale(reference):
    """An untrained model at loss scale 65536 overflows fp16 in some gradients with plain PyTorch too (the trainer's scaler
    would skip that step). The kernels must overflow in the same parameters and no others."""
    m, ids, labels = reference["model"], reference["ids"], reference["labels"]
    kernels.configure(liger=False, xformers=False)
    plain = overflowing(m, ids, labels, 65536.0)
    kernels.configure(liger=True, xformers=True)
    fused = overflowing(m, ids, labels, 65536.0)
    print(f"\n  loss scale 65536: {len(plain)} parameters overflow in PyTorch, {len(fused)} with the kernels")
    assert fused <= plain, f"new overflow in {sorted(fused - plain)}"


def test_embedding_gradient_with_heavily_repeated_rows():
    """Liger's embedding backward adds into the table's own dtype (fp16) with atomics; F.embedding sums each row in fp32.
    Zipf-skewed ids make a few rows collect thousands of terms, the case where that difference would show."""
    from budgie.kernels import embedding
    torch.manual_seed(5)
    N, V, d = 8192, 2048, 128
    ids = (torch.rand(N, device=DEV) ** 6 * V).long().clamp_max(V - 1)      # row 0 gets a large share
    table = torch.randn(V, d, device=DEV, dtype=torch.float16)
    g = (torch.randn(N, d, device=DEV, dtype=torch.float16) * 0.05)
    truth = torch.zeros(V, d, device=DEV, dtype=torch.float32).index_add_(0, ids, g.float())
    errors = {}
    for name, fn in (("F.embedding", lambda w: torch.nn.functional.embedding(ids, w)), ("Liger", lambda w: embedding(ids, w))):
        w = table.clone().requires_grad_()
        fn(w).backward(g)
        errors[name] = rel(w.grad.float(), truth)
    print(f"\n  embedding gradient error vs fp32 sum (most-repeated row {int((ids == 0).sum())} of {N} tokens): {errors}")
    assert errors["Liger"] <= max(2 * errors["F.embedding"], 5e-3), errors


def test_per_token_nll_matches_pytorch(reference):
    m, ids, labels = reference["model"], reference["ids"], reference["labels"]
    out = {}
    for on in (False, True):
        kernels.configure(liger=on)
        with torch.no_grad():
            hidden = m.model(input_ids=ids, use_cache=False).last_hidden_state
            out[on] = m.head.token_nll(hidden, labels, m.model.embed.adaptive)
    assert torch.equal(out[True][1], out[False][1])                       # same ranks
    valid = out[False][1] >= 0
    err = (out[True][0] - out[False][0])[valid].abs().max().item()
    assert err < 2e-3, f"per-token NLL differs by {err}"
    assert out[True][0][~valid].abs().max().item() == 0.0
    print(f"\n  per-token NLL: max difference {err:.1e} over {int(valid.sum())} tokens")


def test_loss_without_adaptive_clusters_and_all_ignored_rows():
    m = make_model(adaptive_cutoffs=())          # one cluster: an ordinary tied full softmax
    ids, labels = batch(ignore=True)
    all_ignored = labels.clone()
    all_ignored[1] = -100                          # a whole sequence ignored
    for name, lab in (("ignored every 13th", labels), ("one sequence ignored", all_ignored)):
        losses = {}
        for on in (False, True):
            kernels.configure(liger=on)
            losses[on] = run(m, ids, lab)[0].item()
        assert abs(losses[True] - losses[False]) < 2e-3, f"{name}: {losses}"


def test_rms_norm_and_rope_leave_their_inputs_alone():
    from budgie.norms import QKNorm, RMSNorm
    from budgie.rotary import RotaryEmbedding
    kernels.configure(liger=True)
    x = torch.randn(2, 50, 4, 128, device=DEV, dtype=torch.float16)
    before = x.clone()
    norm = QKNorm(128).to(DEV, torch.float16)
    y = norm(x)
    assert y.dtype == torch.float16 and torch.equal(x, before)
    cos, sin = RotaryEmbedding(128, 10000.0)(torch.arange(50, device=DEV)[None])
    z = RotaryEmbedding.apply(y.transpose(1, 2), cos, sin)
    assert torch.equal(x, before) and z.shape == (2, 4, 50, 128) and z.dtype == torch.float16
    res = torch.randn(2, 50, 256, device=DEV)                                  # the fp32 residual stream
    out = RMSNorm(256).to(DEV, torch.float16)(res)
    assert out.dtype == torch.float16 and out.shape == res.shape


@pytest.mark.parametrize("window,tail", [(None, False), (48, False), (None, True)])
def test_xformers_attention_equals_the_aten_kernel(window, tail):
    from budgie.sdpa import sdpa_window
    B, H, T, D = 2, 4, 200, 128
    torch.manual_seed(3)
    q, k, v = (torch.randn(B, H, T, D, device=DEV, dtype=torch.float16) for _ in range(3))
    sinks, g = torch.randn(H, device=DEV), torch.randn(B, H, T, D, device=DEV, dtype=torch.float16)
    got = {}
    for on in (False, True):
        kernels.configure(attention=on)
        a, b, c, s = (t.clone().requires_grad_() for t in (q, k, v, sinks))
        out = sdpa_window(a, b, c, D ** -0.5, s, window, tail=tail)
        out.backward(g)
        got[on] = (out.detach(), a.grad, b.grad, c.grad, s.grad)
    for label, x, y in zip(("output", "dq", "dk", "dv", "dsink"), got[True], got[False]):
        assert rel(x.float(), y.float()) < 5e-3, f"{label}: {rel(x.float(), y.float())}"
    print(f"\n  window {window} tail {tail}: output max |diff| {(got[True][0] - got[False][0]).abs().max().item():.1e}")


def test_kernels_survive_torch_compile_without_new_graph_breaks():
    """The trainer compiles every layer (config.COMPILE_LAYERS); each kernel is a custom op, so it must not add graph breaks."""
    from torch._dynamo.utils import counters
    breaks = {}
    for on in (False, True):
        kernels.configure(liger=on, xformers=on)
        torch._dynamo.reset()
        counters["graph_break"].clear()
        m = make_model()
        for layer in m.model.layers:
            layer.compile()
        ids, labels = batch()
        loss, grads = run(m, ids, labels)
        breaks[on] = sum(counters["graph_break"].values())
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads.values())
        if on:
            ref_loss = loss
        else:
            plain = loss
    assert abs(ref_loss.item() - plain.item()) < 5e-3
    assert breaks[True] <= breaks[False], f"graph breaks: {breaks[False]} without the kernels, {breaks[True]} with"
    print(f"\n  compiled layers: graph breaks {breaks[False]} without, {breaks[True]} with the kernels; loss {plain.item():.5f} vs {ref_loss.item():.5f}")


def test_cached_decoding_and_generate_are_unchanged():
    kernels.configure(liger=False, xformers=False)
    m = make_model().eval()
    ids, _ = batch(B=2, T=40, ignore=False)
    outs = {}
    for on in (False, True):
        kernels.configure(liger=on, xformers=on)
        with torch.no_grad():
            outs[on] = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=24, do_sample=False, pad_token_id=0)
    assert torch.equal(outs[True], outs[False])


def test_self_check_passes_on_this_gpu():
    kernels.configure(liger=True, xformers=True)
    assert kernels.self_check(torch.device(DEV)) == {n: "ok" for n in kernels.FLAGS}


def test_everything_is_off_unless_asked_for(monkeypatch):
    assert not any(AT_IMPORT.values()) or os.environ.get("BUDGIE_KERNELS"), f"kernels on without being asked: {AT_IMPORT}"
    for name in kernels.FLAGS:
        kernels.FLAGS[name] = False
    monkeypatch.delenv("BUDGIE_KERNELS", raising=False)
    kernels._from_environment()
    assert kernels.active() == []
    monkeypatch.setenv("BUDGIE_KERNELS", "rms_norm, xformers")
    kernels._from_environment()
    assert kernels.active() == ["rms_norm", "attention"]
    with pytest.raises(ValueError):
        kernels.configure(swiglu=True)
