"""The files that make this directory a loadable Hugging Face model repo: config.json, generation_config.json, tokenizer/."""

import json
import pathlib

import pytest
from transformers import AutoTokenizer, GenerationConfig

from budgie import BudgieConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
THINK = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(ROOT / "tokenizer")


def test_config_json_loads_and_points_at_files_in_this_repo():
    cfg = BudgieConfig.from_pretrained(ROOT)
    assert cfg.model_type == "budgie" and cfg.num_hidden_layers == len(cfg.layer_pattern) * cfg.num_blocks
    for target in json.loads((ROOT / "config.json").read_text())["auto_map"].values():
        assert (ROOT / (target.split(".")[0] + ".py")).exists(), target


def test_default_layer_structure():
    cfg = BudgieConfig()
    assert cfg.block_pattern == ["S", "D4-16", "D16-64", "D4-16", "D2-4", "SUM", "D32-128", "D64-256", "G_shared"]
    assert cfg.num_hidden_layers == 64 and cfg.has_branch and cfg.branch_pos == 5 and cfg.layer_pattern[cfg.branch_pos] == "D32-128"
    assert (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim) == (1536, 12, 128)
    assert cfg.attention_types["S"]["kv_heads"] == 6 and cfg.lin_rule == "gated_delta" and cfg.adaptive_cutoffs == [16384, 40000]
    assert cfg.attention_types["D2-4"] == {"window": 2048, "dilation": 2, "conv": 2, "rope": True, "kv_heads": 6}
    assert cfg.attention_types["D16-64"] == {"window": 16384, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 4}
    assert cfg.attention_types["D32-128"] == {"window": 32768, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 2}
    assert cfg.attention_types["D64-256"] == {"window": 65536, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 2}
    assert cfg.attention_types["G_shared"] == {"window": None, "dilation": 1, "conv": 4, "rope": False, "kv_heads": 4}
    kinds = cfg.layer_kinds
    readers = {i for i in cfg.kv_share}
    assert {kinds[i] for i in readers} == {"D16-64", "D32-128", "D64-256", "G_shared"}   # S, D2-4 and D4-16 keep their own
    sparse = [i for i in range(cfg.num_hidden_layers) if cfg.sparse_spec(i)]
    assert sparse == [b * 8 + 7 for b in (1, 3, 5, 7)] and all(i in readers for i in sparse)   # blocks 2, 4, 6, 8's global layers
    assert json.loads((ROOT / "config.json").read_text())["block_pattern"] == cfg.block_pattern, "config.json is stale: regenerate it"


def test_generation_config_stops_on_eos_and_end_of_turn(tok):
    gen = GenerationConfig.from_pretrained(ROOT)
    assert gen.eos_token_id == [tok.eos_token_id, tok.convert_tokens_to_ids("<|end|>")]


def test_tokenizer_matches_the_config(tok):
    cfg = BudgieConfig.from_pretrained(ROOT)
    assert len(tok) == cfg.vocab_size
    assert (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) == (cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id)


def test_think_and_tool_tokens_are_single_ids_that_survive_decoding(tok):
    ids = tok.convert_tokens_to_ids(THINK)
    assert ids == list(range(10, 16))
    text = "".join(THINK)
    encoded = tok.encode(text, add_special_tokens=False)
    assert encoded == ids
    assert tok.decode(encoded, skip_special_tokens=True) == text  # generation parsing needs them kept


def test_plain_tokenization_is_bos_text_without_eos(tok):
    bos, eos = tok.bos_token_id, tok.eos_token_id
    for text in ("hi", "a longer sentence, with punctuation.", ""):
        ids = tok(text)["input_ids"]
        assert ids[0] == bos and eos not in ids
    assert tok("hi")["input_ids"] == [bos, *tok("hi", add_special_tokens=False)["input_ids"]]
    pair = tok("a", "b")["input_ids"]
    assert pair[0] == bos and eos not in pair
    assert all(b[0] == bos and eos not in b for b in tok(["x", "yz"])["input_ids"])
    assert tok("hi" + tok.eos_token)["input_ids"][-1] == eos      # an <eos> written in the text is still one token


def test_model_max_length_matches_the_config(tok):
    assert tok.model_max_length == BudgieConfig.from_pretrained(ROOT).max_position_embeddings == 1048576
    assert len(tok("word " * 3000)["input_ids"]) > 1024


def test_chat_template_tokenize_true_is_unchanged_and_has_no_eos(tok):
    msgs = [{"role": "user", "content": "hi"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    assert ids[:8] == [1, 4, 230, 5184, 6, 230, 5, 230] and ids[8:] == tok.encode("<think>\n", add_special_tokens=False)
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    conv = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "u2"}]
    for kw in (dict(), dict(enable_thinking=False), dict(tools=tools)):
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **kw)
        enc = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, **kw)
        got = enc["input_ids"] if hasattr(enc, "keys") else enc
        assert got == tok.encode(text, add_special_tokens=False)    # tokenized without special tokens, so the post-processor is not involved
        assert got.count(tok.bos_token_id) == 1 and tok.eos_token_id not in got


def test_chat_template_without_extras_is_the_plain_format(tok):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "again"}]
    got = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert got == "<bos><|system|>\nS<|end|>\n<|user|>\nhi<|end|>\n<|assistant|>\nyo<|end|>\n<|user|>\nagain<|end|>\n<|assistant|>\n<think>\n"


def test_thinking_is_always_on(tok):
    msgs = [{"role": "user", "content": "hi"}]
    outs = {tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            for kw in (dict(), dict(enable_thinking=True), dict(enable_thinking=False))}
    assert outs == {"<bos><|user|>\nhi<|end|>\n<|assistant|>\n<think>\n"}           # no toggle
    done = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "yo"}], tokenize=False)
    assert done.endswith("<|assistant|>\n<think>\n\n</think>\n\nyo<|end|>\n")         # no reasoning: an empty block


def test_reasoning_is_kept_only_after_the_last_user_turn(tok):
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "old"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2", "reasoning_content": "new"}]
    out = tok.apply_chat_template(msgs, tokenize=False)
    assert "old" not in out and "<think>\nnew\n</think>\n\na2" in out
    inline = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>\nthoughts\n</think>\n\nanswer"}]
    assert "<think>\nthoughts\n</think>\n\nanswer<|end|>" in tok.apply_chat_template(inline, tokenize=False)


def test_tool_calls_and_responses(tok):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "w", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    msgs = [{"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                                                                {"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}]},
            {"role": "tool", "content": "18C"}, {"role": "tool", "content": "20C"}]
    out = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    assert out.startswith("<bos><|system|>\n# Tools") and "<tools>\n" in out and '"name": "get_weather"' in out
    assert ('<|assistant|>\n<think>\n\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call><|end|>\n') in out
    assert out.endswith("<|user|>\n<tool_response>\n18C\n</tool_response>\n<tool_response>\n20C\n</tool_response><|end|>\n<|assistant|>\n<think>\n")
