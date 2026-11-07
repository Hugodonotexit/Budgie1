"""The two halves of a decoder layer, computed `chunk` tokens at a time in forward AND backward.

offload.py keeps the per-layer inputs off the GPU, but backward still recomputes a whole half with grad: ~74 MiB
per 1k tokens for attention and ~32 for the FFN (compiled, measured), i.e. 9.5 GiB for attention at 128k, on top
of a base of ~7.6 GiB. Here backward goes chunk by chunk, so that working set is that of one chunk:

    FFN half         x + ffn(norm(x)): chunk c needs the (conv width - 1) tokens before it for the causal conv
                     (they are computed again and dropped); forward and backward are a plain loop over chunks.
    attention half   x + attn(norm(x)). The queries of a chunk attend over K/V of the whole sequence (the keys
                     they can see: AttentionPattern.attend_chunk), and K/V are the one thing that has to exist at
                     full length ([B, kv_heads, T, D], 2 MiB per 1k tokens each). Backward is staged:
                       A  recompute K/V (no grad) into leaf tensors,
                       B  per query chunk: recompute the chunk with grad, backward -> dx, parameter gradients,
                          and dK, dV accumulated in the leaves (starting from the gradient the layers that read
                          this layer's K/V have already produced),
                       C  per chunk, recompute K/V with grad and push the dK, dV slice back -> dx, K/V-path
                          parameter gradients.
                     A layer that READS another layer's K/V (config.kv_share) skips A and C and returns dK, dV.
                     A compressed + selected layer (sparse.py) also pools K/V once for the whole sequence; in B the
                     pooled K/V are leaves too, and their gradient goes back through the pooling (by segments of
                     whole blocks) before dK, dV are used. Its chunks also compute the `group` queries before them,
                     which the first query group's block selection needs.

Both go through _Staged, a custom autograd Function that holds its input in pinned host memory (offload.py).
All of it is exact: the same operations on the same values, only grouped differently.
"""

import torch


_BIG = 1 << 20     # elements: an extra at least this big (a layer's K or V) is kept in host memory like x


class _Staged(torch.autograd.Function):
    @staticmethod
    def forward(ctx, impl, pool, x, *extras):
        ctx.impl, ctx.pool, ctx.device = impl, pool, x.device
        buf = pool.take(x)
        buf.copy_(x, non_blocking=True)
        ctx.buf = buf
        ctx.hosted, small = {}, []
        for i, t in enumerate(extras):
            if t.numel() >= _BIG:
                ctx.hosted[i] = pool.take(t)
                ctx.hosted[i].copy_(t, non_blocking=True)
            else:
                small.append(t)
        ctx.n_extras = len(extras)
        ctx.save_for_backward(*small)
        return impl.forward(x, *extras)

    @staticmethod
    def backward(ctx, *grads):
        small = iter(ctx.saved_tensors)
        extras = [ctx.hosted[i].to(ctx.device, non_blocking=True) if i in ctx.hosted else next(small) for i in range(ctx.n_extras)]
        for buf in ctx.hosted.values():
            ctx.pool.give(buf)
        ctx.hosted = {}
        x = ctx.buf.to(ctx.device, non_blocking=True)
        ctx.pool.give(ctx.buf)
        ctx.buf = None
        dx, dextras = ctx.impl.backward(x, extras, grads)
        return (None, None, dx, *dextras)


def staged(impl, pool, x, *extras):
    """impl.forward(x, *extras) -> tuple of tensors, with x kept in `pool` and impl.backward(x, extras, grads) ->
    (dx, grads of extras) run instead of autograd's own backward."""
    out = _Staged.apply(impl, pool, x, *extras)
    return out if isinstance(out, tuple) else (out,)


def _spans(T, chunk):
    return [(s, min(s + chunk, T)) for s in range(0, T, chunk)]


class FFNHalf:
    def __init__(self, layer, chunk, sample_pos=None):
        self.layer, self.chunk, self.sample_pos = layer, chunk, sample_pos
        self.context = layer.ffn.conv.width - 1

    def _run(self, x_ext, s, lo):
        pos = None if self.sample_pos is None else self.sample_pos[:, lo:lo + x_ext.shape[1]]
        y = self.layer.ffn(self.layer.ffn_norm(x_ext), None, None, pos)[:, s - lo:]
        return x_ext[:, s - lo:] + y

    def forward(self, x):
        out = torch.empty_like(x)
        for s, e in _spans(x.shape[1], self.chunk):
            lo = max(0, s - self.context)
            out[:, s:e] = self._run(x[:, lo:e], s, lo)
        return (out,)

    def backward(self, x, extras, grads):
        g, dx = grads[0], torch.zeros_like(x)
        for s, e in reversed(_spans(x.shape[1], self.chunk)):
            lo = max(0, s - self.context)
            x_ext = x[:, lo:e].detach().requires_grad_(True)
            with torch.enable_grad():
                out = self._run(x_ext, s, lo)
            torch.autograd.backward(out, g[:, s:e])
            dx[:, lo:e] += x_ext.grad
        return dx, ()


_POOL_SEGMENT = 1 << 16   # tokens of K/V pooled at once (sparse layers)


class AttentionHalf:
    def __init__(self, layer, cos_sin, chunk):
        self.layer, self.cos_sin, self.chunk = layer, cos_sin, chunk
        attn = layer.self_attn
        self.owner = attn.owner
        self.sparse = attn.sparse
        self.before = attn.sparse["group"] if attn.sparse is not None else 0
        if self.before and chunk % self.before:
            raise ValueError(f"a sparse layer's chunk ({chunk}) must be a multiple of its query group ({self.before})")
        self.context_q = attn.q_conv.width - 1 + self.before
        self.context_kv = attn.kv_conv.width - 1 if attn.owner else 0

    def _positions(self, lo, e):
        cos, sin = self.cos_sin
        return cos[:, lo:e], sin[:, lo:e]

    def _kv(self, x_ext, s, lo):
        """K, V of the tokens s .. (lo + len(x_ext)) from the context-extended chunk x_ext = x[:, lo:e]."""
        k, v, _ = self.layer.self_attn._own_kv(self.layer.attn_norm(x_ext), self._positions(lo, lo + x_ext.shape[1]), None, None, 0)
        return k[:, :, s - lo:], v[:, :, s - lo:]

    def _attend(self, x_ext, k, v, s, lo, pooled=None):
        attn = self.layer.self_attn
        e = lo + x_ext.shape[1]
        q_all = attn._queries(self.layer.attn_norm(x_ext), self._positions(lo, e), None, None,
                              attn.query_positions(lo, e - lo, x_ext.device))
        q = q_all[:, :, s - lo:]
        if self.sparse is not None:
            q_before = q_all[:, :, s - self.before - lo:s - lo] if s else None
            out, _ = attn.attend_sparse(q, k[:, :, :e], v[:, :, :e], s, kc=pooled[0], vc=pooled[1], q_before=q_before)
        else:
            out = attn.pattern.attend_chunk(q, k, v, s, attn.sinks, attn.scale, attn.n_rep)
        return x_ext[:, s - lo:] + attn.o_proj(out.transpose(1, 2).reshape(x_ext.shape[0], e - s, -1))

    def _all_kv(self, x):
        """K and V for the whole sequence, chunk by chunk, without grad."""
        k = v = None
        with torch.no_grad():
            for s, e in _spans(x.shape[1], self.chunk):
                lo = max(0, s - self.context_kv)
                kk, vv = self._kv(x[:, lo:e], s, lo)
                if k is None:
                    k = torch.empty(kk.shape[0], kk.shape[1], x.shape[1], kk.shape[3], dtype=kk.dtype, device=x.device)
                    v = torch.empty_like(k)
                k[:, :, s:e], v[:, :, s:e] = kk, vv
        return k, v

    def _pool(self, k, v):
        """Pooled K/V of the whole sequence for a sparse layer, without grad, a segment of whole blocks at a time."""
        m = self.sparse["compress"]
        seg = max(m, _POOL_SEGMENT // m * m)
        with torch.no_grad():
            parts = [self.layer.self_attn.compress(k[:, :, a:a + seg], v[:, :, a:a + seg])[:2] for a in range(0, k.shape[2] // m * m, seg)]
        if not parts:
            empty = k.new_zeros(k.shape[0], k.shape[1], 0, k.shape[3], dtype=torch.promote_types(k.dtype, torch.float32))
            return empty, empty.clone()
        return torch.cat([p[0] for p in parts], 2), torch.cat([p[1] for p in parts], 2)

    def _pool_backward(self, k, v, kc, vc):
        """Send the pooled K/V's gradient back through the pooling into k.grad, v.grad and pool_score's gradient."""
        if kc.grad is None and vc.grad is None:
            return
        m = self.sparse["compress"]
        seg = max(m, _POOL_SEGMENT // m * m)
        gk = kc.grad if kc.grad is not None else torch.zeros_like(kc)
        gv = vc.grad if vc.grad is not None else torch.zeros_like(vc)
        for a in range(0, kc.shape[2] * m, seg):
            with torch.enable_grad():
                pk, pv, _ = self.layer.self_attn.compress(k[:, :, a:a + seg], v[:, :, a:a + seg])
            torch.autograd.backward([pk, pv], [gk[:, :, a // m:a // m + pk.shape[2]], gv[:, :, a // m:a // m + pv.shape[2]]])

    def forward(self, x, *shared):
        if self.owner:
            k, v = self._all_kv(x)
            k_pos = torch.arange(x.shape[1], device=x.device)
        else:
            k, v, k_pos = shared
        pooled = self._pool(k, v) if self.sparse is not None else None
        out = torch.empty_like(x)
        for s, e in _spans(x.shape[1], self.chunk):
            lo = max(0, s - self.context_q)
            out[:, s:e] = self._attend(x[:, lo:e], k, v, s, lo, pooled)
        return (out, k, v, k_pos) if self.owner else (out,)

    def backward(self, x, extras, grads):
        g, dx = grads[0], torch.zeros_like(x)
        if self.owner:
            k, v = self._all_kv(x)
            k.requires_grad_(True)
            v.requires_grad_(True)
            if grads[1] is not None:       # what the layers that read this layer's K/V already sent back
                k.grad, v.grad = grads[1].clone(), grads[2].clone()
        else:
            k, v = extras[0].detach().requires_grad_(True), extras[1].detach().requires_grad_(True)
        pooled = None
        if self.sparse is not None:
            pooled = tuple(t.requires_grad_(True) for t in self._pool(k, v))
        for s, e in reversed(_spans(x.shape[1], self.chunk)):                       # B
            lo = max(0, s - self.context_q)
            x_ext = x[:, lo:e].detach().requires_grad_(True)
            with torch.enable_grad():
                out = self._attend(x_ext, k, v, s, lo, pooled)
            torch.autograd.backward(out, g[:, s:e])
            dx[:, lo:e] += x_ext.grad
        if pooled is not None:
            self._pool_backward(k, v, *pooled)
        if not self.owner:
            return dx, (k.grad, v.grad, None)
        for s, e in reversed(_spans(x.shape[1], self.chunk)):                       # C
            lo = max(0, s - self.context_kv)
            x_ext = x[:, lo:e].detach().requires_grad_(True)
            with torch.enable_grad():
                kk, vv = self._kv(x_ext, s, lo)
            torch.autograd.backward([kk, vv], [k.grad[:, :, s:e], v.grad[:, :, s:e]])
            dx[:, lo:e] += x_ext.grad
        return dx, ()


class DocumentAttentionHalf:
    """AttentionHalf with document masking (documents.py): the documents of the rows run as sequences of their own,
    those of one length bucket stacked into a batch and zero-padded at the end (documents.stack_padded), each batch
    through its own AttentionHalf, and the results put back in place. segments: [(row, start, end)] covering every row.
    cos/sin are cut the same way (positions restart at every document when the model made them from document_starts)."""

    def __init__(self, layer, cos_sin, chunk, segments):
        from .documents import by_bucket
        self.owner = layer.self_attn.owner
        self.groups = []
        for L, members in by_bucket(segments).items():
            cs = tuple(_stack(t, members, L, 1, rows_of=t.shape[0] > 1) for t in cos_sin)
            self.groups.append((L, members, AttentionHalf(layer, cs, chunk)))

    def forward(self, x, *shared):
        out = torch.empty_like(x)
        k = v = None
        for L, members, half in self.groups:
            res = half.forward(_stack(x, members, L, 1), *self._shared(shared, members, L))
            _put(out, res[0], members, 1)
            if self.owner:
                if k is None:
                    k = res[1].new_empty(x.shape[0], res[1].shape[1], x.shape[1], res[1].shape[3])
                    v = torch.empty_like(k)
                _put(k, res[1], members, 2)
                _put(v, res[2], members, 2)
        if not self.owner:
            return (out,)
        return out, k, v, torch.arange(x.shape[1], device=x.device)

    @staticmethod
    def _shared(shared, members, L):
        if not shared:
            return ()
        k, v = _stack(shared[0], members, L, 2), _stack(shared[1], members, L, 2)
        return k, v, torch.arange(L, device=k.device)

    def backward(self, x, extras, grads):
        dx = torch.zeros_like(x)
        dk = dv = None
        for L, members, half in self.groups:
            seg = (_stack(grads[0], members, L, 1),)
            if self.owner:
                seg += tuple(None if g is None else _stack(g, members, L, 2) for g in grads[1:3])
            d, dext = half.backward(_stack(x, members, L, 1), self._shared(extras, members, L), seg)
            _put(dx, d, members, 1)
            if not self.owner:
                if dk is None:
                    dk, dv = torch.zeros_like(extras[0]), torch.zeros_like(extras[1])
                _put(dk, dext[0], members, 2)
                _put(dv, dext[1], members, 2)
        if not self.owner:
            return dx, (dk, dv, None)
        return dx, ()


def _stack(t, members, L, dim, rows_of=True):
    """documents.stack_padded of t (T at `dim`); rows_of False: t has one row that every member shares (cos/sin)."""
    from .documents import stack_padded
    if not rows_of:
        members = [(0, s, e) for _, s, e in members]
    return stack_padded(t, members, L, dim)


def _put(dst, src, members, dim):
    """Write each member's first (end - start) positions of src [n, ..., L, ...] into dst at its place."""
    for i, (b, s, e) in enumerate(members):
        dst[b].narrow(dim - 1, s, e - s).copy_(src[i].narrow(dim - 1, 0, e - s))
