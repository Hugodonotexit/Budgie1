"""CUDA tests of the graph-replayed decode path (decode.py): it must match the eager cached path on a tiny
configuration whose windows are short enough that the ring buffers wrap. Skipped without a GPU."""

import pytest
import torch

from budgie import BudgieCache
from test_model import tiny_model, tokens

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode graph needs CUDA")


def _decode(model, ids, head, fast, extra=0):
    from budgie import decode
    decode.ENABLED = fast
    try:
        with torch.no_grad():
            cache = BudgieCache(model.config)
            out = [model(ids[:, :head], past_key_values=cache, use_cache=True).logits]
            out += [model(ids[:, t:t + 1], past_key_values=cache, use_cache=True).logits for t in range(head, ids.shape[1])]
            if extra:  # an ordinary multi-token call after fast decoding hands the state back to the cache
                tail = ids[:, -extra:]
                out.append(model(tail, past_key_values=cache, use_cache=True).logits)
                nxt = ids[:, :1]
                out += [model(nxt, past_key_values=cache, use_cache=True).logits for _ in range(4)]
        return torch.cat(out, 1), cache
    finally:
        decode.ENABLED = True


@pytest.mark.parametrize("branch", [True, False])
@pytest.mark.parametrize("batch", [1, 3])
def test_fast_decode_matches_eager_decode(branch, batch):
    model = tiny_model(branch).cuda()
    ids = tokens(batch, 60, bos_at=(0, 23)).cuda()  # 35 decoded tokens: every windowed ring wraps
    eager, _ = _decode(model, ids, 25, fast=False)
    fast, cache = _decode(model, ids, 25, fast=True)
    assert torch.allclose(eager, fast, atol=2e-4)
    assert cache.fast_engine is not None and cache.get_seq_length() == 60


def test_state_returns_to_the_cache_for_a_later_multi_token_call():
    model = tiny_model(True).cuda()
    ids = tokens(2, 40, bos_at=(0, 17)).cuda()
    eager, _ = _decode(model, ids, 25, fast=False, extra=7)
    fast, cache = _decode(model, ids, 25, fast=True, extra=7)
    assert torch.allclose(eager, fast, atol=2e-4)


def test_generate_matches_with_and_without_the_graph():
    from budgie import decode
    model = tiny_model(True).cuda()
    ids = tokens(2, 12).cuda()
    kw = dict(max_new_tokens=40, do_sample=False, pad_token_id=0, eos_token_id=None)
    with torch.no_grad():
        decode.ENABLED = True
        fast = model.generate(ids, **kw)
        decode.ENABLED = False
        eager = model.generate(ids, **kw)
        decode.ENABLED = True
    assert torch.equal(fast, eager)


def test_padded_batches_stay_on_the_eager_path():
    model = tiny_model(True).cuda()
    ids = tokens(2, 12).cuda()
    mask = torch.ones_like(ids)
    mask[1, :4] = 0
    kw = dict(attention_mask=mask, max_new_tokens=10, do_sample=False, pad_token_id=0, eos_token_id=None, return_dict_in_generate=True)
    with torch.no_grad():
        out = model.generate(ids, **kw)
    assert out.past_key_values.no_fast and out.past_key_values.fast_engine is None
