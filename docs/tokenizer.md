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
| 6 | `<\|end\|>` | 16 | `<file_sep>` |
| | | 17 | `<repo_name>` |
| | | 18–31 | `<reserved_15>` … `<reserved_28>` |
| 7 | `<fim_prefix>` | | |
| 8 | `<fim_suffix>` | | |
| 9 | `<fim_middle>` | | |

* `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>` and `</tool_response>` are **not**
  flagged `special`. They are still single tokens, but `skip_special_tokens=True` keeps them, so generated
  reasoning and tool calls can be parsed from decoded text. To make them special, set `"special": true` on
  their entries in `tokenizer.json` and list them in `special_tokens_map.json`.
* The reserved ids were renamed in place, so the vocabulary size and the embedding shape did not change.
  `<file_sep>` and `<repo_name>` (2026-11-03, before any training) mark files and the repository name in
  repository-level code, as in StarCoder2; `<fim_prefix>`, `<fim_suffix>` and `<fim_middle>` are the
  fill-in-the-middle sentinels.
* **Plain `tok(text)` returns `<bos> text`**, with no trailing `<eos>`: the post-processor adds `<bos>` only (for a
  pair, `tok(a, b)` is `<bos> a b`). So a prompt can be continued as it is, and benchmark harnesses that tokenize
  with special tokens on (lm-evaluation-harness with `add_bos_token=True`) give every context and continuation a
  `<bos>` and no `<eos>`. Turning BOS off in a harness is not a substitute: the model expects `<bos>` at the start of
  a document, and the linear branch resets its state there.
* **A training document is `<bos> text <eos>`**, and the tokenizer no longer writes the `<eos>` for you: append
  `tok.eos_token_id` yourself (`tok(text)["input_ids"] + [tok.eos_token_id]`, or `[bos, *ids, eos]` from
  `add_special_tokens=False`).
* The chat template writes `<bos>` itself, and `apply_chat_template` tokenizes without adding specials, so it is
  not affected by the post-processor.
* `model_max_length` is 524288, the configuration's `max_position_embeddings`.
* A `tokenizer.json` saved before this change (the post-processor template was `<bos> $A <eos>`) still appends
  `<eos>`. Delete the `<eos>` entries from `post_processor.single` and `post_processor.pair` to bring it in line.

## Chat template

Turns are `<|role|>\n{content}<|end|>\n`. A prompt ends with `<|assistant|>\n<think>\n`, and the model ends its turn with
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

Thinking is always on, with no toggle (2026-11-03): the generation prompt ends with `<think>\n`, so the model always
starts in its reasoning block and decides how long it is. `enable_thinking` is ignored.

Assistant messages in the history may carry `reasoning_content`, or a `<think>…</think>` block inside `content`.
Every assistant turn after the last user message is rendered as `<think>\n{reasoning}\n</think>\n\n{answer}`, with
an empty block (`<think>\n\n</think>\n\n`) when there is no reasoning, so training data without reasoning teaches
the model to close the block at once. Earlier turns lose their reasoning.

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

Apart from the `<think>` block of the turns after the last user message and of the generation prompt, the output
without `tools` is the plain system/user/assistant template.
