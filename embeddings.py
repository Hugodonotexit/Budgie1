"""Input embeddings: an adaptive (frequency-clustered) token table plus hashed n-grams."""

import torch
from torch import nn

from .cache import BudgieCache
from .configuration_budgie import NGRAM_HASH_BASES, BudgieConfig


class AdaptiveInput(nn.Module):
    """Token embeddings by frequency rank, in clusters of shrinking width (Baevski & Auli).

    Token ids are first mapped to a rank (`rank_of`, identity until set_vocab_order). The ranks are
    cut into clusters at config.adaptive_cutoffs; cluster i has embedding width
    hidden_size // adaptive_div**i and, for i > 0, a Linear up to hidden_size. Frequent tokens get
    wide embeddings, rare ones narrow. The output head reuses these same tables and projections
    (AdaptiveSoftmaxHead), so this module owns the weights of both ends of the model."""

    def __init__(self, config: BudgieConfig):
        super().__init__()
        self.cuts = config.adaptive_cuts
        self.register_buffer("rank_of", torch.arange(config.vocab_size))
        widths = [config.hidden_size // config.adaptive_div**i for i in range(len(self.cuts) - 1)]
        self.tok = nn.ModuleList([nn.Embedding(self.cuts[i + 1] - self.cuts[i], w) for i, w in enumerate(widths)])
        self.proj = nn.ModuleList([nn.Linear(w, config.hidden_size, bias=False) for w in widths[1:]])

    @torch.no_grad()
    def set_vocab_order(self, order):
        """order[r] = the token id of frequency rank r (most frequent first), a permutation of the vocabulary."""
        order = torch.as_tensor(order, dtype=torch.long, device=self.rank_of.device)
        if order.numel() != self.rank_of.numel() or not torch.equal(order.sort().values, torch.arange(order.numel(), device=order.device)):
            raise ValueError("order must be a permutation of range(vocab_size)")
        self.rank_of[order] = torch.arange(order.numel(), device=order.device)

    def lookup(self, rank):
        """Embeddings of tokens given by frequency rank. Every cluster is looked up for every token
        (their widths make that cheap) and the right one selected, so shapes never depend on the data."""
        out = None
        for i, table in enumerate(self.tok):
            lo, hi = self.cuts[i], self.cuts[i + 1]
            e = table((rank - lo).clamp(0, hi - lo - 1))
            if i > 0:
                e = torch.where(((rank >= lo) & (rank < hi))[..., None], self.proj[i - 1](e), out)
            out = e
        return out

    def forward(self, ids):
        return self.lookup(self.rank_of[ids])


class HashedNgramEmbedding(nn.Module):
    """Hashed n-gram embeddings, added to the residual stream.

    For each order n there is one table of `ngram_buckets` narrow rows (`ngram_dim` wide), split
    into `ngram_heads` equal slices, one per head. The last n token ids are hashed by `ngram_heads`
    independent hash functions, head k into slice k, and the rows found are concatenated and
    projected up to hidden_size by a learned Linear. Slices are not shared between heads, so a row
    is always read through the same slice of the projection. The same parameter budget as a few
    wide rows buys many more buckets, and a collision under one hash is undone by the others.
    Positions before the start of the text see the pad id as history, exactly as left padding does."""

    def __init__(self, config: BudgieConfig):
        super().__init__()
        self.orders, self.buckets, self.pad_id = config.ngram_orders, config.ngram_buckets, config.pad_token_id
        self.heads, dim = config.ngram_heads, config.ngram_dim
        self.rows_per_head = self.buckets // self.heads
        self.tables = nn.ModuleList([nn.Embedding(self.buckets, dim) for _ in self.orders])
        self.proj = nn.ModuleList([nn.Linear(self.heads * dim, config.hidden_size, bias=False) for _ in self.orders])

    @staticmethod
    def _mix(h):
        """Avalanche a 31-bit hash (the murmur3 finalizer, cut to 31 bits): every output bit depends on
        every input bit. Without it the low bits of a polynomial hash keep lattice structure, and taking
        them mod a power of two collides badly for n-grams that differ in one token only."""
        mask = 0x7FFFFFFF
        h = h ^ (h >> 15)
        h = (h * 0x2C1B3C6D) & mask
        h = h ^ (h >> 12)
        h = (h * 0x297A2D39) & mask
        return h ^ (h >> 15)

    def raw_hashes(self, ctx, n, keep, T):
        """The hash of the n-gram ending at each of the T positions, per head: [B, T, heads], in
        [0, 2^31). `ctx` is the previous `keep` token ids followed by the T current ones. Each head
        is a polynomial hash mod 2^31 - 1 with its own multiplier, then mixed (all products stay below
        2^61, so int64 never overflows)."""
        out = []
        for base in NGRAM_HASH_BASES[: self.heads]:
            h = torch.full_like(ctx[:, keep:], 7919 * n)
            for j in range(n):  # current token, then one further back each time
                h = (h * base + ctx[:, keep - j : keep - j + T] + 1) % 2147483647
            out.append(self._mix(h))
        return torch.stack(out, -1)

    def bucket_ids(self, ctx, n, keep, T):
        """Row indices [B, T, heads]: head k's hash taken mod the slice size and offset into slice k."""
        offsets = torch.arange(self.heads, device=ctx.device) * self.rows_per_head
        return self.raw_hashes(ctx, n, keep, T) % self.rows_per_head + offsets

    def forward(self, x, ids, cache: BudgieCache | None = None):
        """Returns x plus the n-gram embeddings of `ids` [B, T]. With a cache, the previous tokens
        come from it and it is updated, so decoding one token at a time matches a full forward."""
        if not self.orders:
            return x
        keep, (B, T) = max(self.orders) - 1, ids.shape
        hist = cache.last_ids if cache is not None and cache.last_ids is not None else ids.new_full((B, keep), self.pad_id)
        ctx = torch.cat([hist, ids], 1)  # the previous `keep` ids, then these
        for n, table, proj in zip(self.orders, self.tables, self.proj):
            rows = table(self.bucket_ids(ctx, n, keep, T))  # [B, T, heads, dim]
            x = x + proj(rows.reshape(B, T, -1))
        if cache is not None and keep:
            cache.last_ids = ctx[:, -keep:].clone()
        return x


class BudgieEmbedding(nn.Module):
    """AdaptiveInput plus HashedNgramEmbedding: the model's input."""

    def __init__(self, config: BudgieConfig):
        super().__init__()
        self.fp32_residual = config.fp32_residual
        self.adaptive = AdaptiveInput(config)
        self.ngram = HashedNgramEmbedding(config)

    def set_vocab_order(self, order):
        self.adaptive.set_vocab_order(order)

    def forward(self, ids, cache: BudgieCache | None = None):
        x = self.adaptive(ids)
        if self.fp32_residual:
            x = x.float()
        return self.ngram(x, ids, cache)
