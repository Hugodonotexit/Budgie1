"""CPU tests of the model on a tiny configuration: shapes, causality, cached decoding, generate, save/load."""

import json
import subprocess
import sys

import pytest
import torch

from budgie import BudgieCache, BudgieConfig, BudgieForCausalLM, BudgieModel
from budgie.configuration_budgie import SHARED_KINDS

BOS = 1
VOCAB = 300
# Small windows, so that the caches really do drop tokens within a short test sequence.
TYPES = {
    "S": {"window": 6, "dilation": 1, "conv": 2, "rope": True, "kv_heads": 2},
    "D1-4": {"window": 4, "dilation": 3, "conv": 3, "rope": True, "kv_heads": 2},
    "D4-16": {"window": 5, "dilation": 2, "conv": 4, "rope": True, "kv_heads": 2},
    "D16-64": {"window": 7, "dilation": 2, "conv": 4, "rope": True, "kv_heads": 2},
    "G_shared": {"window": None, "dilation": 1, "conv": 2, "rope": False, "kv_heads": 4},
}
BRANCH = dict(lin_heads=2, lin_head_dim=8, lin_conv_kernel=4, lin_chunk=8,
              lin_half_life_range=(4, 64), lin_gate_init=0.5, lin_head_gate=True)


# Small blocks, so the 2nd block's compressed + selected global layer really pools and selects in short sequences.
SPARSE = {"every": 2, "compress": 4, "select": 2, "group": 3, "local": 5}


def tiny_config(branch=True, **kw):
    base = dict(vocab_size=VOCAB, hidden_size=32, num_attention_heads=4, head_dim=16, attention_types=TYPES, num_blocks=2,
                intermediate_size=(48, 48, 48), adaptive_cutoffs=(20, 80), ngram_buckets=64, max_position_embeddings=512,
                sparse_global=SPARSE,
                block_pattern=("S", "D1-4", "S", "D4-16", "D16-64", "SUM", "G_shared") if branch else ("S", "D1-4", "S", "D4-16", "D16-64", "G_shared"))
    base.update(BRANCH if branch else {})
    base.update(kw)
    return BudgieConfig(**base)


def tiny_model(branch=True, seed=0, **kw):
    torch.manual_seed(seed)
    model = BudgieForCausalLM(tiny_config(branch, **kw))
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():  # identity convs and zero sinks would hide wiring mistakes
        for name, p in model.named_parameters():
            if "conv" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif "sinks" in name or "gain" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3 + (1.0 if "gain" in name else 0.0))
            elif "cluster_vectors" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    model.model.embed.set_vocab_order(torch.randperm(VOCAB, generator=g))
    return model.eval()


def tokens(B, T, seed=2, bos_at=()):
    ids = torch.randint(3, VOCAB, (B, T), generator=torch.Generator().manual_seed(seed))
    for p in bos_at:
        ids[:, p] = BOS
    return ids


@pytest.mark.parametrize("branch", [True, False])
def test_forward_returns_log_probabilities(branch):
    model = tiny_model(branch)
    with torch.no_grad():
        out = model(tokens(2, 24), use_cache=False)
    assert out.logits.shape == (2, 24, VOCAB)
    assert torch.allclose(out.logits.exp().sum(-1), torch.ones(2, 24), atol=1e-4)


@pytest.mark.parametrize("branch", [True, False])
def test_loss_and_gradients(branch):
    model = tiny_model(branch).train()
    ids = tokens(2, 24)
    out = model(ids, labels=ids, use_cache=False)
    assert out.logits is None and torch.isfinite(out.loss)
    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached {missing}"


@pytest.mark.parametrize("branch", [True, False])
def test_causal(branch):
    model = tiny_model(branch)
    a, b = tokens(1, 30, seed=3), tokens(1, 30, seed=4)
    b[:, :20] = a[:, :20]
    with torch.no_grad():
        la, lb = model(a, use_cache=False).logits, model(b, use_cache=False).logits
    assert torch.allclose(la[:, :20], lb[:, :20], atol=1e-5)
    assert not torch.allclose(la[:, 20:], lb[:, 20:], atol=1e-3)


@pytest.mark.parametrize("branch", [True, False])
def test_prefill_then_decode_equals_full_forward(branch):
    model = tiny_model(branch)
    ids = tokens(2, 40, bos_at=(0, 17))
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        cache = BudgieCache(model.config)
        head = model(ids[:, :25], past_key_values=cache, use_cache=True).logits
        steps = [model(ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits for t in range(25, 40)]
    assert torch.allclose(head, full[:, :25], atol=1e-4)
    assert torch.allclose(torch.cat(steps, 1), full[:, 25:], atol=1e-4)


def test_generate_matches_uncached_generate():
    model = tiny_model(True)
    ids = tokens(2, 12)
    kw = dict(max_new_tokens=16, do_sample=False, pad_token_id=0, eos_token_id=None)
    with torch.no_grad():
        cached = model.generate(ids, use_cache=True, **kw)
        plain = model.generate(ids, use_cache=False, **kw)
    assert cached.shape == (2, 28)
    assert torch.equal(cached, plain)


def test_base_model_returns_hidden_states():
    model = BudgieModel(tiny_config()).eval()
    with torch.no_grad():
        out = model(tokens(2, 10), use_cache=False)
    assert out.last_hidden_state.shape == (2, 10, 32)


def test_retention_branch_off_adds_no_parameters():
    on, off = tiny_model(True), tiny_model(False)
    assert sum(p.numel() for p in on.parameters()) > sum(p.numel() for p in off.parameters())
    assert off.model.branches is None
    assert "lin_heads" not in off.config.to_dict() and not off.config.has_branch and on.config.has_branch
    assert BudgieConfig.from_dict({**tiny_config(True).to_dict(), "block_pattern": ["S", "D1-4", "S", "D4-16", "D16-64", "G_shared"]}).has_branch is False


def test_default_kv_sharing_pairs_consecutive_blocks():
    cfg = tiny_config(num_blocks=4)
    size = len(cfg.layer_pattern)
    shared = [p for p, kind in enumerate(cfg.layer_pattern) if kind in SHARED_KINDS]
    assert shared == [4, 5]                 # D16-64, G_shared; S, D1-4 and D4-16 always compute their own
    assert cfg.kv_owner(size + 4) == 4      # block 2's D16-64 reads block 1's
    assert cfg.kv_owner(size + 0) is None and cfg.kv_owner(size + 3) is None
    assert cfg.kv_owner(3 * size + 5) == 2 * size + 5  # G_shared
    assert sorted(cfg.kv_share) == sorted(b * size + p for b in (1, 3) for p in shared)
    assert tiny_config(num_blocks=4, kv_share={}).kv_share == {}


@pytest.mark.parametrize("kw", [
    dict(block_pattern=("S", "X")),
    dict(num_attention_heads=3),
    dict(adaptive_cutoffs=(80, 20)),
    dict(adaptive_cutoffs=(20, VOCAB)),
    dict(ngram_heads=3),
    dict(block_pattern=("S", "D1-4", "SUM")),
    dict(kv_share={0: 1}),
])
def test_invalid_configs_are_rejected(kw):
    with pytest.raises(ValueError):
        tiny_config(**kw)


def test_save_and_reload(tmp_path):
    model = tiny_model(True)
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["auto_map"]["AutoModelForCausalLM"] == "modeling_budgie.BudgieForCausalLM"
    assert (tmp_path / "modeling_budgie.py").exists(), "the model code is saved next to the weights"

    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    ids = tokens(2, 20)
    with torch.no_grad():
        assert torch.allclose(model(ids, use_cache=False).logits, reloaded(ids, use_cache=False).logits, atol=1e-6)

    # A fresh interpreter that has never imported this package: only trust_remote_code can find the classes.
    code = (
        "import sys, torch\n"
        "from transformers import AutoModelForCausalLM\n"
        f"m = AutoModelForCausalLM.from_pretrained({str(tmp_path)!r}, trust_remote_code=True).eval()\n"
        f"ids = torch.load({str(tmp_path / 'ids.pt')!r})\n"
        "with torch.no_grad():\n"
        "    torch.save(m(ids, use_cache=False).logits, sys.argv[1])\n"
    )
    torch.save(ids, tmp_path / "ids.pt")
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "out.pt")], check=True, cwd=tmp_path, capture_output=True)
    with torch.no_grad():
        assert torch.allclose(model(ids, use_cache=False).logits, torch.load(tmp_path / "out.pt"), atol=1e-6)


def test_rope_extension_yarn_and_linear():
    import math
    from budgie.rotary import RotaryEmbedding
    base = RotaryEmbedding(128, 5e5)
    inv, m = base.inv_freq("cpu")
    assert m == 1.0 and base.scaling is None
    lin, _ = RotaryEmbedding(128, 5e5, {"type": "linear", "factor": 4.0}).inv_freq("cpu")
    assert torch.allclose(lin, inv / 4)
    yarn = RotaryEmbedding(128, 5e5, {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144})
    inv_y, ms = yarn.inv_freq("cpu")
    turns = 262144 * inv / (2 * math.pi)
    assert torch.equal(inv_y[turns >= 32], inv[turns >= 32])                     # fast dimensions untouched
    assert torch.allclose(inv_y[turns <= 1], inv[turns <= 1] / 4)                # slow ones interpolated
    assert abs(ms - (0.1 * math.log(4) + 1)) < 1e-12
    cos, sin = yarn(torch.arange(8)[None])
    assert torch.allclose(cos[0, 0], torch.full((128,), ms))                     # position 0: cos = mscale


def test_rope_extension_config_round_trip_and_validation(tmp_path):
    ext = {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}
    cfg = tiny_config(rope_extension=ext)
    cfg.save_pretrained(tmp_path)
    back = BudgieConfig.from_pretrained(tmp_path)
    assert back.rope_extension == ext and tiny_config().rope_extension is None
    assert BudgieForCausalLM(back).model.rotary.scaling == ext
    for bad in ({"type": "ntk", "factor": 4.0}, {"type": "yarn", "factor": 4.0}, {"type": "linear", "factor": 0.5}):
        with pytest.raises(ValueError):
            tiny_config(rope_extension=bad)


def test_rope_angles_are_exact_at_one_million():
    """The angle is formed in fp64 and reduced mod 2 pi before the fp32 cast: cos/sin match an fp64 reference to fp32
    output rounding at positions up to 2^20 (a plain fp32 product is off by ~0.03 rad there)."""
    import math
    from budgie.rotary import RotaryEmbedding
    for scaling in (None, {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}):
        rot = RotaryEmbedding(128, 5e5, scaling)
        pos = torch.arange(2**20 - 2048, 2**20)[None]
        cos, sin = rot(pos)
        inv, m = rot.inv_freq("cpu")
        ref = pos[0, :, None].double() * inv[None]
        assert (cos[0, :, :64].double() - m * ref.cos()).abs().max() < 1e-6
        assert (sin[0, :, 64:].double() - m * ref.sin()).abs().max() < 1e-6
        old = pos[0, :, None].float() * inv.float()[None]                       # the fp32 product it replaced
        assert (old.double().cos() - ref.cos()).abs().max() > 1e-2


def test_rope_angle_chunks_change_nothing(monkeypatch):
    from budgie.rotary import RotaryEmbedding
    rot = RotaryEmbedding(16, 1e4, {"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 64})
    pos = torch.arange(3, 3 + 2 * 101).view(2, 101)
    whole = rot(pos)
    monkeypatch.setattr(RotaryEmbedding, "ANGLE_CHUNK", 7)
    parts = rot(pos)
    assert all(torch.equal(a, b) for a, b in zip(whole, parts))


def test_global_logit_scale_is_off_below_the_trained_length_and_consistent_when_on():
    ids = tokens(2, 40, bos_at=(0, 17))
    with torch.no_grad():
        base = tiny_model(True)(ids, use_cache=False).logits
        long_l = tiny_model(True, global_logit_scale={"type": "log", "trained_length": 40})(ids, use_cache=False).logits
        assert torch.equal(base, long_l)                                         # every position < 40: factor exactly 1
        model = tiny_model(True, global_logit_scale={"type": "log", "trained_length": 8})
        on = model(ids, use_cache=False).logits
        assert torch.equal(on[:, :7], base[:, :7]) and not torch.allclose(on[:, 20:], base[:, 20:], atol=1e-3)
        cache = BudgieCache(model.config)
        head = model(ids[:, :25], past_key_values=cache, use_cache=True).logits
        steps = [model(ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits for t in range(25, 40)]
    assert torch.allclose(head, on[:, :25], atol=1e-4)
    assert torch.allclose(torch.cat(steps, 1), on[:, 25:], atol=1e-4)
    scaled = [l.self_attn.logit_scale_length for l in model.model.layers]
    assert {s for s, k in zip(scaled, model.config.layer_kinds) if model.config.attention_types[k]["window"] is None} == {8}
    assert {s for s, k in zip(scaled, model.config.layer_kinds) if model.config.attention_types[k]["window"] is not None} == {None}


def test_global_logit_scale_setter_round_trip_and_validation(tmp_path):
    model = tiny_model(True)
    assert "global_logit_scale" not in model.config.to_dict()
    model.model.set_global_logit_scale({"type": "log", "trained_length": 8})
    model.save_pretrained(tmp_path)
    back = BudgieForCausalLM.from_pretrained(tmp_path).eval()
    ids = tokens(1, 30)
    with torch.no_grad():
        assert torch.allclose(model(ids, use_cache=False).logits, back(ids, use_cache=False).logits, atol=1e-6)
    assert back.config.global_logit_scale == {"type": "log", "trained_length": 8}
    model.model.set_global_logit_scale(None)
    assert all(l.self_attn.logit_scale_length is None for l in model.model.layers)
    for bad in ({"type": "linear", "trained_length": 8}, {"type": "log"}, {"type": "log", "trained_length": 1}):
        with pytest.raises(ValueError):
            tiny_config(global_logit_scale=bad)
