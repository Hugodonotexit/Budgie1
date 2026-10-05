# Architecture

All numbers are the defaults of `BudgieConfig()` unless stated. Everything is configurable; see the
[configuration reference](#configuration-reference).

![Budgie structure](structure.svg)

The diagram is generated from `config.json` by `docs/make_structure_diagram.py` (standard library only); re-run it
after changing the defaults.

## Layout of the stack

A **block** is a sequence of layer kinds, `block_pattern = (S, D4, S, D8, D16, G)`, and the model is
`num_blocks` blocks (8 → 48 layers; `num_hidden_layers` is derived and cannot be set).

| Kind | Window | Dilation | Reach | Rotary | KV heads | Conv kernel |
|---|---|---|---|---|---|---|
| `S` local | 4096 | 1 | 4k | yes | 8 | 3 |
| `D4` dilated | 4096 | 4 | 16k | yes | 4 | 4 |
| `D8` dilated | 8192 | 4 | 32k | yes | 4 | 4 |
| `D16` dilated | 16384 | 4 | 64k | yes | 8 | 4 |
| `G` global | all | 1 | all | no | 8 | 4 |

A query at position *i* in a dilated layer sees *i, i−d, i−2d, …* (*window* tokens). Dilation is implemented
by dealing the tokens into *d* interleaved sequences, each an ordinary windowed problem, so a dilated layer
costs the same as a local one. `G` has no positional encoding, and the whole model relies on the local layers'
rotary positions and convolutions for order. `attention_types` redefines the kinds, and `block_pattern` can use
any letters defined there.

Every layer is

```
x = x + attention(rms_norm(x))
x = x + ffn(rms_norm(x))
```

with the residual stream kept in fp32 (`fp32_residual`) and a final RMSNorm before the head.

## Attention

* Q reads a depthwise **causal convolution** of the normed input; K and V read a second one. The kernel is
  2–4 taps (per kind). With `causal-conv1d` installed and on CUDA the convolution uses its kernel (wrapped as a
  `torch.library` custom op so `torch.compile` does not break on it); otherwise `F.conv1d`.
* **QK-norm**: queries and keys are RMS-normalised per head over `head_dim`, with a learned scale, before rotary.
* **Sinks**: one learned logit per head acts as an extra softmax slot with a zero value, so a head can attend to
  "nothing". On CUDA the sink is merged with the memory-efficient kernel's own sliding-window mode; elsewhere it
  is an extra key column.
* **Grouped-query attention**: `kv_heads` per kind. K/V are expanded to the full head count before SDPA
  (`enable_gqa` falls back to the slow math kernel on some GPUs).
* **KV sharing** (`kv_share`): a map from a *reader* layer to an earlier *owner* layer of the same kind. A reader
  keeps its own Q projection, output projection and FFN, and has no K/V projection, K/V convolution, K/V norm or
  cache. The default makes every second block read the previous block's `D4`, `D8`, `D16` and `G` K/V (every kind
  except `S`). `{}` turns it off.

## Feed-forward

A depthwise causal convolution (`ffn_conv_kernel`), then `n + 1` bias-free linear layers
`d → w₁ → … → wₙ → d`, where `intermediate_size = [w₁ … wₙ]`. The default `[2048, 2048, 2048]` at `d = 1024`
gives four matrices. Activations: SiLU after each hidden layer except the middle one (`n // 2`), which uses the
**centred dSiLU**: the derivative of SiLU, shifted so that it is exactly 0 at 0 (its zero crossing at x ≈ −1.278 is
moved to the origin). Its range is (−0.0998, 1.0998).

`intermediate_size` may also be a list of such lists (one per block, or one per layer) to vary the FFN with depth.

## Embeddings and output

* **Adaptive input.** Token ids are mapped to a frequency rank by the buffer `embed.rank_of`, then embedded by
  cluster: ranks below 2048 get width *d*, ranks 2048–8191 get *d/2*, the rest *d/4*, the narrower ones projected
  up by a Linear. **`rank_of` is the identity until `model.model.embed.set_vocab_order(order)` is called** (with the
  token ids sorted most frequent first); trained checkpoints carry it. It has to match the tokenizer and the data
  the model was trained on.
* **Hashed n-grams.** For each order in `ngram_orders` (2, 3) the previous n tokens are hashed by `ngram_heads` (2)
  independent hash functions, each owning its slice of a table of `ngram_buckets` (262,144) narrow rows
  (`ngram_dim = d/8`). The heads' rows are concatenated and projected up to *d*. This is added to the token
  embedding. The cache keeps the last `max(order) − 1` ids so decoding matches a full forward.
* **Output head.** An adaptive softmax tied to the input tables: the first cluster's tokens are scored against its
  embedding rows plus one logit per tail cluster (`head.cluster_vectors`), and tail tokens are scored inside their
  cluster through the same projection used on the input side. Logits are soft-capped at `logit_softcap` (50).
  `adaptive_cutoffs=[]` gives an ordinary tied full softmax. The loss is computed in checkpointed chunks of 2048
  tokens, so full-vocabulary logits are never materialised for a whole batch.

## Retention branch

With `linear_branch="before_G"`, each block has one RetNet-style linear-attention branch that reads the residual
stream at the start of the block (own RMSNorm and causal convolution), runs in parallel with the block's local layers
and adds a gated output just before the block's first `G` layer. Per head, with a fixed decay
`γ = exp(−ln 2 / half_life)`:

```
S_t = γ S_{t−1} + (1 − γ) k_t v_tᵀ        (fp32, reset at every <bos>)
o_t = q_tᵀ S_t / (1 − γ^n)                n = tokens since the document start
```

Each branch has `lin_heads` (16) heads of `lin_head_dim` (64). Half-lives are spread log-uniformly over
`lin_half_life_range` (256 … 32768 tokens) and staggered from block to block, so the blocks together cover the
range more densely; `q` and `k` are L2-normalised,
and the output is multiplied by a learned scalar `g` (initially `lin_gate_init = 0.05`) and, if `lin_head_gate`, a
per-token per-head sigmoid. The recurrence is evaluated `lin_chunk` tokens at a time; `retention_reference` in
`retention.py` is the token-by-token fp64 oracle. `"after_G"` is reserved but not implemented. With
`linear_branch="none"` the model has no branch modules and `config.json` carries no `lin_*` keys; a `config.json`
without `linear_branch` loads as `"none"`.

## Cache

`BudgieCache` holds, per layer, the keys and values that a later query can still see (`(window − 1) · dilation`
past tokens; everything for `G`; nothing for a KV-sharing reader) and the states of the causal convolutions (Q, FFN,
K/V, and the branch's). The retention branch's fp32 state lives in the block's first `G` layer; `last_doc_pos` tracks
the position within the current document. A `BudgieCache` is created automatically; a generic `DynamicCache` is not
accepted.

## Configuration reference

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | 42000 | must match the tokenizer |
| `hidden_size` | 1024 | *d* |
| `block_pattern` | `S D4 S D8 D16 G` | layer kinds of one block |
| `num_blocks` | 8 | blocks in the stack |
| `attention_types` | see above | per kind: `window`, `dilation`, `conv`, `rope`, `kv_heads` |
| `num_attention_heads`, `head_dim` | 16, 128 | `kv_heads` of every kind must divide the head count |
| `kv_share` | every 2nd block reads the one before | reader layer → owner layer |
| `qk_norm`, `attention_sinks` | true, true | |
| `intermediate_size` | `[2048, 2048, 2048]` | FFN hidden widths (int, list, or list of lists) |
| `ffn_hidden_layers` | 3 | used when `intermediate_size` is an int |
| `ffn_conv_kernel` | 4 | |
| `ngram_orders`, `ngram_buckets`, `ngram_dim`, `ngram_heads` | (2, 3), 262144, d/8, 2 | hashed n-gram input |
| `adaptive_cutoffs`, `adaptive_div` | (2048, 8192), 2 | frequency clusters and width ratio; `[]` = full softmax |
| `logit_softcap` | 50.0 | 0/None disables |
| `linear_branch` | `before_G` | `none` / `before_G` |
| `lin_heads`, `lin_head_dim`, `lin_conv_kernel`, `lin_chunk` | 16, 64, 4, 256 | branch shape |
| `lin_base_heads` | none | set when a trained branch was widened: its first `lin_base_heads` heads keep their original half-lives and the other heads take the positions in between (`retention.branch_half_lives`). Omitted from `config.json` when none |
| `lin_half_life_range`, `lin_gate_init`, `lin_head_gate` | (256, 32768), 0.05, true | branch decay and gating |
| `rms_norm_eps`, `rope_theta` | 1e-5, 10000 | |
| `max_position_embeddings` | 65536 | not enforced; rotary has no table |
| `fp32_residual` | true | residual stream in fp32 |
| `initializer_range` | 0.025 | |
| `tie_word_embeddings` | true | the head always reuses the input tables |

Invalid combinations (unknown layer kinds, heads not divisible by `kv_heads`, unsorted `adaptive_cutoffs`, a
`kv_share` owner that is not an earlier layer of the same kind, …) raise `ValueError` at construction.
