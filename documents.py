"""Document masking: packed documents that do not see each other.

A row of training data can hold several documents back to back (image + caption pairs, chats). Given
`document_starts` [B, T] (True at the first token of each document), BudgieModel runs every document as if it
were alone in the row:

    attention        each layer attends within each document only, dilated and compressed + selected layers too:
                     the documents are padded at their end to a few length buckets and stacked, and each bucket
                     runs the layer's ordinary attention as one batch (per_document);
    causal convs     a conv tap that would reach back past the document's first token reads zero;
    n-gram input     the n-grams of a document's first tokens see the pad id as history;
    retention        the branch's state starts from zero at every document start, as it does at <bos>;
    RoPE             positions count from 0 at every document start.

So the output for a document equals what the model gives for that document alone (up to float rounding). Rows without a start after position 0 are not touched, and a call whose
document_starts mark nothing after position 0 is an ordinary call.

`sample_pos` [B, T] (documents.positions) is what the layers get: the 1-based position of each token within
its document. The document boundaries the attention needs (segments) are read back to the host once per
forward and kept for that tensor.
"""

import weakref
from collections import defaultdict

import torch

from .retention import document_positions

_SEGMENTS = {}   # (data_ptr, shape, stride) of a sample_pos -> (weak reference to the tensor it came from, segments)


def positions(document_starts):
    """[B, T] bool -> sample_pos [B, T] int64: 1 at each document's first token (and at position 0), counting up."""
    return document_positions(document_starts.bool())


def segments(sample_pos):
    """[(row, start, end)] of every document of sample_pos [B, T], row by row, in order. Each row is covered from
    0 to T. Computed once per tensor (one read back to the host) and cached while the tensor lives; a view of it
    (detach(), as a checkpoint's recompute passes) finds the same entry, since the memory cannot be reused while
    the tensor it was computed from is alive."""
    key = (sample_pos.data_ptr(), tuple(sample_pos.shape), sample_pos.stride(), str(sample_pos.device))
    hit = _SEGMENTS.get(key)
    if hit is not None and hit[0]() is not None:
        return hit[1]
    B, T = sample_pos.shape
    starts = [[] for _ in range(B)]
    for b, t in (sample_pos == 1).nonzero().tolist():
        starts[b].append(t)
    segs = []
    for b, ss in enumerate(starts):
        if not ss or ss[0] != 0:
            raise ValueError(f"row {b}: sample_pos must be 1 at position 0")
        segs += [(b, s, e) for s, e in zip(ss, ss[1:] + [T])]
    _SEGMENTS[key] = (weakref.ref(sample_pos, lambda _, k=key: _SEGMENTS.pop(k, None)), segs)
    return segs


def has_boundaries(document_starts):
    """Whether any row has a document start after position 0."""
    return document_starts is not None and document_starts.shape[1] > 1 and bool(document_starts[:, 1:].any())


def bucket(length):
    """The padded length a document of `length` tokens is batched at: rounded up to 1/8 of its power of two (at least
    16), so documents of similar lengths share one call and the padding is at most 12.5%."""
    q = max(16, (1 << (max(1, length).bit_length() - 1)) // 8)
    return -(-length // q) * q


def by_bucket(segs):
    """{padded length: [(row, start, end)]} of the segments (bucket)."""
    groups = defaultdict(list)
    for b, s, e in segs:
        groups[bucket(e - s)].append((b, s, e))
    return groups


def stack_padded(t, members, L, dim):
    """t [B, ..., T, ...] -> [n, ..., L, ...]: each member's slice (row b, positions s..e) zero-padded at its end to L.
    Padding at the end of a document is in its future, which causal attention, convs and pooling never read."""
    parts = []
    for b, s, e in members:
        x = t[b].narrow(dim - 1, s, e - s)
        if e - s < L:
            pad = [0, 0] * (x.dim() - dim) + [0, L - (e - s)]
            x = torch.nn.functional.pad(x, pad)
        parts.append(x)
    return torch.stack(parts)


def per_document(fn, sample_pos, *tensors, dim=2):
    """fn applied to the documents of sample_pos, batched: every tensor [B, ..., T, ...] (T at `dim`) is cut into its
    documents, documents of one bucket are zero-padded at the end to its length and stacked, and fn(*batched) -> one
    tensor [n, ..., L, ...] whose first (document length) positions are kept. Returns that output put back together,
    [B, ..., T, ...]. fn must be causal (a position never reads a later one), so the padding changes nothing."""
    segs = segments(sample_pos)
    B, T = sample_pos.shape
    pieces = {}
    for L, members in by_bucket(segs).items():
        if all(s == 0 and e == T for _, s, e in members):            # whole rows: no copy beyond picking them
            rows = torch.tensor([b for b, _, _ in members], device=tensors[0].device)
            batch = [t.index_select(0, rows) for t in tensors]
        else:
            batch = [stack_padded(t, members, L, dim) for t in tensors]
        out = fn(*batch)
        for i, (b, s, e) in enumerate(members):
            pieces[b, s] = out[i].narrow(dim - 1, 0, e - s)
    rows = []
    for b in range(B):
        parts = [pieces[b, s] for bb, s, _ in segs if bb == b]
        rows.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim - 1))
    return torch.stack(rows)
