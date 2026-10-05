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
    assert cfg.model_type == "budgie" and cfg.num_hidden_layers == len(cfg.block_pattern) * cfg.num_blocks
    for target in json.loads((ROOT / "config.json").read_text())["auto_map"].values():
        assert (ROOT / (target.split(".")[0] + ".py")).exists(), target


def test_default_layer_structure():
    cfg = BudgieConfig()
    assert cfg.block_pattern == ["S", "D4", "S", "D8", "D16", "G"] and cfg.num_hidden_layers == 48
    assert cfg.attention_types["S"]["window"] == 4096 and cfg.attention_types["D4"]["window"] == 4096
    assert cfg.attention_types["D8"] == {"window": 8192, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 4}
    assert cfg.attention_types["D16"] == {"window": 16384, "dilation": 4, "conv": 4, "rope": True, "kv_heads": 8}
    kinds = cfg.layer_kinds
    readers = {i for i in cfg.kv_share}
    assert all(kinds[i] != "S" for i in readers) and {kinds[i] for i in readers} == {"D4", "D8", "D16", "G"}
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


def test_chat_template_without_extras_is_the_plain_format(tok):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}, {"role": "user", "content": "again"}]
    got = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert got == "<bos><|system|>\nS<|end|>\n<|user|>\nhi<|end|>\n<|assistant|>\nyo<|end|>\n<|user|>\nagain<|end|>\n<|assistant|>\n"


def test_enable_thinking_toggle(tok):
    msgs = [{"role": "user", "content": "hi"}]
    on = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    on_explicit = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert on == on_explicit and on.endswith("<|assistant|>\n")
    assert off == on + "<think>\n\n</think>\n\n"


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
    assert ('<|assistant|>\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call><|end|>\n') in out
    assert out.endswith("<|user|>\n<tool_response>\n18C\n</tool_response>\n<tool_response>\n20C\n</tool_response><|end|>\n<|assistant|>\n")
