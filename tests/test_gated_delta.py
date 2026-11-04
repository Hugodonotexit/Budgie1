"""CPU tests of the Gated DeltaNet branch (lin_rule="gated_delta"): the chunked recurrence against its token-by-token
fp64 reference, document resets, a state carried across calls, cached decoding, segmented forward, initialization
and configuration loading."""

import inspect
import json

import torch
from torch.nn import functional as F

from budgie import BudgieConfig
from budgie.retention import document_positions, gated_delta_chunked, gated_delta_reference
from test_model import BOS, tiny_config, tiny_model, tokens


def qkv(B, T, H, dk, dv, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, dk, generator=g, dtype=torch.float64), dim=-1)
    k = F.normalize(torch.randn(B, T, H, dk, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(B, T, H, dv, generator=g, dtype=torch.float64)
    lam = torch.rand(B, T, H, generator=g, dtype=torch.float64) * 0.3       # half-lives of a few tokens: decay matters
    beta = torch.rand(B, T, H, generator=g, dtype=torch.float64)
    return q, k, v, lam, beta


def doc_pos(B, T, bos_at, seed=1):
    ids = torch.randint(3, 100, (B, T), generator=torch.Generator().manual_seed(seed))
    for b, positions in enumerate(bos_at):
        for p in positions:
            ids[b, p] = BOS
    return document_positions(ids == BOS)


def gd_model(seed=0):
    """Decay and write projections randomized, so the per-token decay and beta really vary (a_proj starts at 0)."""
    model = tiny_model(seed=seed, lin_rule="gated_delta")
    g = torch.Generator().manual_seed(seed + 13)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".a_proj." in name or ".b_proj." in name or name.endswith(".dt_bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.7)
    return model


def test_chunked_matches_reference():
    B, T, H = 3, 150, 4
    q, k, v, lam, beta = qkv(B, T, H, 16, 8)
    worst = 0.0
    for bos_at in ([[], [], []], [[64, 128], [100], [64, 100]], [[0, 1, 2], [0], list(range(2, T, 3))]):
        dp = doc_pos(B, T, bos_at)
        ref_o, ref_S = gated_delta_reference(q, k, v, lam, beta, dp)
        for chunk in (1, 7, 64, 150, 256):
            o, S = gated_delta_chunked(q, k, v, lam, beta, dp, None, chunk)
            worst = max(worst, (o - ref_o).abs().max().item(), (S - ref_S).abs().max().item())
    assert worst <= 1e-12, worst


def test_state_carried_across_calls_and_packed_documents():
    B, T, H = 3, 150, 4
    q, k, v, lam, beta = qkv(B, T, H, 16, 8, seed=4)
    dp = doc_pos(B, T, [[30], [], [90]])
    ref_o, ref_S = gated_delta_reference(q, k, v, lam, beta, dp)
    o1, S1 = gated_delta_chunked(q[:, :70], k[:, :70], v[:, :70], lam[:, :70], beta[:, :70], dp[:, :70], None, 16)
    o2, S2 = gated_delta_chunked(q[:, 70:], k[:, 70:], v[:, 70:], lam[:, 70:], beta[:, 70:], dp[:, 70:], S1, 16)
    assert (torch.cat([o1, o2], 1) - ref_o).abs().max() <= 1e-12 and (S2 - ref_S).abs().max() <= 1e-12
    # a document starting mid-chunk sees nothing of the one before it
    start = 90
    dp_ab = doc_pos(B, T, [[0, 31, start], [0, start], [0, start]])
    o_ab, _ = gated_delta_chunked(q, k, v, lam, beta, dp_ab, None, 64)
    sl = slice(start, None)
    o_b, _ = gated_delta_chunked(q[:, sl], k[:, sl], v[:, sl], lam[:, sl], beta[:, sl], dp_ab[:, sl], None, 64)
    assert (o_ab[:, sl] - o_b).abs().max() <= 1e-12


def test_fp32_long_chunk_is_accurate():
    T = 2100
    q, k, v, _, beta = qkv(1, T, 2, 32, 32, seed=3)
    lam = torch.full((1, T, 2), 1e-5, dtype=torch.float64)                   # slow decay: the state keeps everything
    dp = document_positions(torch.zeros(1, T, dtype=torch.bool))
    ref_o, _ = gated_delta_reference(q, k, v, lam, beta, dp)
    o, S = gated_delta_chunked(q.float(), k.float(), v.float(), lam.float(), beta.float(), dp, None, 1024)
    assert (o.double() - ref_o).abs().max() / ref_o.abs().max() <= 1e-4 and torch.isfinite(S).all()


def test_prefill_then_decode_equals_full_forward():
    ids = tokens(2, 44, bos_at=(12, 31))
    model = gd_model()
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits
        out = model(input_ids=ids[:, :25], use_cache=True)
        cache, got = out.past_key_values, [out.logits]
        for t in range(25, ids.shape[1]):
            out = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            got.append(out.logits)
    assert (torch.cat(got, 1) - full).abs().max() <= 1e-5


def test_segmented_equals_whole_sequence():
    branch = gd_model().model.branches[0]
    x0 = torch.randn(2, 70, 32, generator=torch.Generator().manual_seed(4))
    dp = document_positions(tokens(2, 70, bos_at=(20, 50)) == BOS)
    with torch.no_grad():
        assert (branch(x0, None, dp, None) - branch.forward_segmented(x0, dp, 16)).abs().max() <= 1e-5


def test_init_and_gradients():
    for branch in tiny_model(lin_rule="gated_delta").model.branches:
        assert branch.a_proj.weight.abs().max() == 0 and branch.dt_bias.abs().max() == 0 and branch.b_proj.bias.abs().max() == 0
    model = gd_model()
    model.train()
    ids = tokens(2, 40, bos_at=(10, 25))
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    grads = {n: p.grad.abs().max().item() for n, p in model.named_parameters() if ".branches." in n}
    assert sum(n.endswith(("a_proj.weight", "b_proj.weight", "b_proj.bias", "dt_bias")) for n in grads) == 2 * 4
    assert all(torch.isfinite(torch.tensor(g)) and g > 0 for g in grads.values()), grads


def test_config_default_old_configs_and_round_trip(tmp_path):
    assert inspect.signature(BudgieConfig.__init__).parameters["lin_rule"].default == "gated_delta"
    try:
        tiny_config(lin_rule="mamba")
        raise AssertionError("accepted lin_rule='mamba'")
    except ValueError:
        pass
    old = tiny_config(lin_rule="gated_delta").to_dict()
    del old["lin_rule"]                                                      # a config.json from before lin_rule
    assert BudgieConfig.from_dict(old).lin_rule == "retention"
    tiny_config(lin_rule="gated_delta").save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["lin_rule"] == "gated_delta"
    assert BudgieConfig.from_pretrained(tmp_path).lin_rule == "gated_delta"
    tiny_config(branch=False).save_pretrained(tmp_path)
    assert "lin_rule" not in json.loads((tmp_path / "config.json").read_text())
