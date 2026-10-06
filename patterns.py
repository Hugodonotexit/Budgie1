"""Which earlier tokens a query may attend to: local, dilated or global."""

import torch
import torch.nn.functional as F

from .sdpa import repeat_kv, sdpa, sdpa_window, window_attention_supported


class AttentionPattern:
    """The attention pattern of one layer kind: a query at position i sees the `window` tokens
    i, i - d, i - 2d, ... (d = `dilation`), or every earlier token when `window` is None.

    A pattern owns everything that depends on those two numbers: whether a query may see a key
    (`allowed`), how many past tokens a cache must keep (`keep_past`), and the attention itself
    in two forms -- `attend` for a whole sequence from position 0, whose cost grows as
    T * window instead of T^2, and `attend_masked` for anything else (a cache, a padded batch)."""

    def __init__(self, window, dilation=1):
        if window is not None and window < 2:
            raise ValueError(f"window must be >= 2 or None, got {window}")
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        self.window, self.dilation = window, dilation

    @classmethod
    def from_spec(cls, spec):
        """From one entry of BudgieConfig.attention_types."""
        return cls(spec["window"], spec["dilation"])

    def __repr__(self):
        return f"AttentionPattern(window={self.window}, dilation={self.dilation})"

    @property
    def is_global(self):
        return self.window is None

    @property
    def reach(self):
        """How far back a query can see, in tokens (None = unbounded)."""
        return None if self.is_global else self.window * self.dilation

    @property
    def keep_past(self):
        """How many past tokens a KV cache must hold for the next query (None = all of them)."""
        return None if self.is_global else (self.window - 1) * self.dilation

    def allowed(self, q_pos, k_pos):
        """Bool [len(q_pos), len(k_pos)]: True where the query at q_pos may attend to the key at k_pos."""
        rel = q_pos[:, None] - k_pos[None, :]
        ok = rel >= 0
        if self.dilation > 1:
            ok = ok & (rel % self.dilation == 0)
        if not self.is_global:
            ok = ok & (rel // self.dilation < self.window)
        return ok

    def attend(self, q, k, v, sinks, scale):
        """q, k, v: [B, H, T, D] for tokens 0 .. T-1, nothing padded. With dilation d the tokens are
        dealt into d interleaved sequences (position mod d), each an ordinary windowed problem, so a
        dilated layer costs the same as an undilated one."""
        d = self.dilation
        if d == 1:
            return self._causal_window(q, k, v, sinks, scale)
        B, H, T, D = q.shape
        pad = (-T) % d
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))  # future positions: never seen by a real query
        L = (T + pad) // d
        fold = lambda t: t.reshape(B, H, L, d, D).permute(0, 3, 1, 2, 4).reshape(B * d, H, L, D)
        out = self._causal_window(fold(q), fold(k), fold(v), sinks, scale)
        return out.reshape(B, d, H, L, D).permute(0, 2, 3, 1, 4).reshape(B, H, T + pad, D)[:, :, :T]

    def attend_chunk(self, q, k, v, start, sinks, scale, n_rep):
        """The queries at positions start .. start + Tq - 1 of a whole sequence, attending over that sequence's
        K/V: q [B, H, Tq, D]; k, v [B, kv_heads, T, D] (not expanded). Equals attend() on the whole sequence
        restricted to those queries, so a long sequence can be done a chunk at a time with only K/V at full
        length. With dilation d, `start` and Tq must be multiples of d (and so is the end of the keys it reads,
        which is `start + Tq`). The key slice is expanded to the query heads one K/V head at a time (n_rep query
        heads each): all at once it is [B, H, keys, D] twice, which for a global layer at 128k is 1 GiB."""
        Tq, d = q.shape[2], self.dilation
        end = start + Tq
        lo = 0 if self.window is None else max(0, start - (self.window - 1) * d)
        lo -= lo % d
        if not window_attention_supported(q, sinks):
            return self.attend_masked(q, repeat_kv(k[:, :, lo:end], n_rep), repeat_kv(v[:, :, lo:end], n_rep),
                                      torch.arange(start, end, device=q.device), torch.arange(lo, end, device=q.device), None, sinks, scale)
        if d > 1 and (start % d or Tq % d):
            raise ValueError(f"a dilation-{d} chunk must start and end on a multiple of {d} (start {start}, length {Tq})")
        outs = []
        for g in range(k.shape[1]):
            heads = slice(g * n_rep, (g + 1) * n_rep)
            outs.append(self._chunk_group(q[:, heads], repeat_kv(k[:, g:g + 1, lo:end], n_rep), repeat_kv(v[:, g:g + 1, lo:end], n_rep),
                                          (start - lo) // d, sinks[heads], scale))
        return torch.cat(outs, 1)

    def _chunk_group(self, q, kc, vc, pad, sinks, scale):
        """One group of heads of attend_chunk; kc, vc are the key slice [lo, end) already expanded to q's heads."""
        d = self.dilation
        B, H, Tq, D = q.shape
        if d > 1:
            fold = lambda t: t.reshape(B, H, t.shape[2] // d, d, D).permute(0, 3, 1, 2, 4).reshape(B * d, H, t.shape[2] // d, D)
            q, kc, vc = fold(q), fold(kc), fold(vc)
        if self.window is None:
            out = sdpa_window(q, kc, vc, scale, sinks, None, tail=True)
        else:
            # The kernel's backward is wrong for a window together with a bottom-right-aligned mask (measured against a
            # float32 reference: relative error ~1 in dq, dk and dv; the forward is right). So the queries are padded
            # with `pad` empty rows in front, which lines the diagonal up under an ordinary top-left mask.
            out = sdpa_window(torch.nn.functional.pad(q, (0, 0, pad, 0)), kc, vc, scale, sinks, self.window)[:, :, pad:]
        if d > 1:
            out = out.reshape(B, d, H, Tq // d, D).permute(0, 2, 3, 1, 4).reshape(B, H, Tq, D)
        return out

    def attend_masked(self, q, k, v, q_pos, k_pos, key_ok, sinks, scale):
        """The same attention from an explicit mask built out of absolute positions. key_ok [B, Tk]
        marks real (non-pad) keys, or is None."""
        allow = self.allowed(q_pos, k_pos)[None, None]
        if key_ok is not None:
            allow = allow & key_ok[:, None, None, :]
        return sdpa(q, k, v, scale, sinks, mask=allow)

    def _causal_window(self, q, k, v, sinks, scale):
        """Undilated: token i sees i - window + 1 .. i (a global pattern: all of 0 .. i). Queries are
        taken `window` at a time and each block only reads the keys it can see (at most 2 * window - 1).
        On CUDA with a sink, the kernel's own sliding window does the same without masks (sdpa_window)."""
        T, window = q.shape[2], self.window
        if window_attention_supported(q, sinks):
            return sdpa_window(q, k, v, scale, sinks, window if window is not None and T > window else None)
        if window is None or T <= window:
            return sdpa(q, k, v, scale, sinks, causal=True)
        pos = torch.arange(T, device=q.device)
        blocks = []
        for start in range(0, T, window):
            end, first = min(start + window, T), max(0, start - window + 1)
            q_pos, k_pos = pos[start:end, None], pos[None, first:end]
            visible = (k_pos <= q_pos) & (q_pos - k_pos < window)
            blocks.append(sdpa(q[:, :, start:end], k[:, :, first:end], v[:, :, first:end], scale, sinks, mask=visible[None, None]))
        return torch.cat(blocks, dim=2)
