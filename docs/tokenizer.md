# Tokenizer and chat format

`tokenizer/` is a BPE tokenizer with 42,000 tokens. Its first 32 ids are special or reserved:

| id | token | id | token |
|---|---|---|---|
| 0 | `<pad>` | 10 | `<think>` |
| 1 | `<bos>` | 11 | `</think>` |
| 2 | `<eos>` | 12 | `<tool_call>` |
| 3 | `<\|system\|>` | 13 | `</tool_call>` |
| 4 | `<\|user\|>` | 14 | `<tool_response>` |
| 5 | `<\|assistant\|>` | 15 | `</tool_response>` |
| 6 | `<\|end\|>` | 16–31 | `<reserved_13>` … `<reserved_28>` |
| 7 | `<fim_prefix>` | | |
| 8 | `<fim_suffix>` | | |
| 9 | `<fim_middle>` | | |

* `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>` and `</tool_response>` are **not**
  flagged `special`. They are still single tokens, but `skip_special_tokens=True` keeps them, so generated
  reasoning and tool calls can be parsed from decoded text. To make them special, set `"special": true` on
  their entries in `tokenizer.json` and list them in `special_tokens_map.json`.
* The reserved ids were renamed in place, so the vocabulary size and the embedding shape did not change.
* **Plain `tok(text)` returns `<bos> text`**, with no trailing `<eos>`: the post-processor adds `<bos>` only (for a
  pair, `tok(a, b)` is `<bos> a b`). So a prompt can be continued as it is, and benchmark harnesses that tokenize
  with special tokens on (lm-evaluation-harness with `add_bos_token=True`) give every context and continuation a
  `<bos>` and no `<eos>`. Turning BOS off in a harness is not a substitute: the model expects `<bos>` at the start of
  a document, and the retention branch resets its state there.
* **A training document is `<bos> text <eos>`**, and the tokenizer no longer writes the `<eos>` for you: append
  `tok.eos_token_id` yourself (`tok(text)["input_ids"] + [tok.eos_token_id]`, or `[bos, *ids, eos]` from
  `add_special_tokens=False`).
* The chat template writes `<bos>` itself, and `apply_chat_template` tokenizes without adding specials, so it is
  not affected by the post-processor.
* `model_max_length` is 65536, the configuration's `max_position_embeddings`; the old value, 1024, made every longer
  text warn.
* A `tokenizer.json` saved before this change (the post-processor template was `<bos> $A <eos>`) still appends
  `<eos>`. Delete the `<eos>` entries from `post_processor.single` and `post_processor.pair` to bring it in line.

## Chat template

Turns are `<|role|>\n{content}<|end|>\n`. A prompt ends with `<|assistant|>\n`, and the model ends its turn with
`<|end|>` (`generation_config.json` stops on it).

```
<bos><|system|>
You are helpful.<|end|>
<|user|>
Hello<|end|>
<|assistant|>
Hi!<|end|>
```

### Thinking

`apply_chat_template(..., enable_thinking=...)`:

* default / `True`: nothing is added; the model decides whether to open `<think>`.
* `False`: the generation prompt ends with `<think>\n\n</think>\n\n`, an empty thinking block, so the model answers directly.

Assistant messages in the history may carry `reasoning_content`, or a `<think>…</think>` block inside `content`.
Only the reasoning of turns after the last user message is kept, and it is rendered as
`<think>\n{reasoning}\n</think>\n\n{answer}`. Earlier turns lose it.

### Tools

With `tools=[...]` (JSON-schema function definitions), the system message gets a `# Tools` section listing them
inside `<tools></tools>`, followed by the call format. The model answers with one `<tool_call>` block per call:

```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
```

An assistant message with `tool_calls` is rendered the same way (arguments may be a dict or a JSON string). Results
come back as `tool` messages; consecutive ones share one user turn:

```
<|user|>
<tool_response>
18C
</tool_response>
<tool_response>
20C
</tool_response><|end|>
```

This is the Hermes/Qwen convention, so existing tool-call parsers for it apply.

Without `tools`, `enable_thinking` and `reasoning_content`, the output is byte-identical to the original
system/user/assistant template.
