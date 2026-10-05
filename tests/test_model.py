"""CPU tests of the model on a tiny configuration: shapes, causality, cached decoding, generate, save/load."""

import json
import subprocess
import sys

import pytest
import torch

from budgie import BudgieCache, BudgieConfig, BudgieForCausalLM, BudgieModel

BOS = 1
VOCAB = 300
# Small windows, so that the caches really do drop tokens within a short test sequence.
TYPES = {
    "S": {"window": 6, "dilation": 1, "conv": 2, "rope": True, "kv_heads": 2},
    "D4": {"window": 4, "dilation": 3, "conv": 3, "rope": True, "kv_heads": 2},
    "D8": {"window": 5, "dilation": 2, "conv": 4, "rope": True, "kv_heads": 2},
    "D16": {"window": 7, "dilation": 2, "conv": 4, "rope": True, "kv_heads": 2},
    "G": {"window": None, "dilation": 1, "conv": 2, "rope": False, "kv_heads": 4},
}
BRANCH = dict(linear_branch="before_G", lin_heads=2, lin_head_dim=8, lin_conv_kernel=4, lin_chunk=8,
              lin_half_life_range=(4, 64), lin_gate_init=0.5, lin_head_gate=True)


def tiny_config(branch=True, **kw):
    base = dict(vocab_size=VOCAB, hidden_size=32, num_attention_heads=4, head_dim=16, attention_types=TYPES, num_blocks=2,
                intermediate_size=(48, 48, 48), adaptive_cutoffs=(20, 80), ngram_buckets=64, max_position_embeddings=512)
    base.update(BRANCH if branch else dict(linear_branch="none"))
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
    assert "linear_branch" not in off.config.to_dict() and "lin_heads" not in off.config.to_dict()
    assert BudgieConfig.from_dict({k: v for k, v in tiny_config(True).to_dict().items() if k != "linear_branch"}).linear_branch == "none"


def test_default_kv_sharing_pairs_consecutive_blocks():
    cfg = tiny_config(num_blocks=4)
    size = len(cfg.block_pattern)
    assert cfg.kv_owner(1 * size + 1) == 1  # block 2's D4 reads block 1's D4
    assert cfg.kv_owner(size + 0) is None   # S layers always compute their own
    assert cfg.kv_owner(size + 2) is None
    assert cfg.kv_owner(3 * size + 4) == 2 * size + 4  # D16
    assert cfg.kv_owner(3 * size + 5) == 2 * size + 5  # G
    assert sorted(cfg.kv_share) == [size + 1, size + 3, size + 4, size + 5, 3 * size + 1, 3 * size + 3, 3 * size + 4, 3 * size + 5]
    assert tiny_config(num_blocks=4, kv_share={}).kv_share == {}


@pytest.mark.parametrize("kw", [
    dict(block_pattern=("S", "X")),
    dict(num_attention_heads=3),
    dict(adaptive_cutoffs=(80, 20)),
    dict(adaptive_cutoffs=(20, VOCAB)),
    dict(ngram_heads=3),
    dict(linear_branch="sideways"),
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
