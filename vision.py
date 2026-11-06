"""Image input: Budgie's own vision transformer (trained from scratch with the language model), a pixel shuffle and
an MLP projector into the residual stream.

Images keep their own aspect ratio and (within a token budget) their own resolution, as in Qwen3-VL / Qwen3.5: the
processor resizes each to a multiple of 32 px a side (patch 16 x pixel_shuffle 2), so one LM token covers 32 x 32 px.
The tower reads an image as a row-major list of 16 x 16 x 3 patches (`pixel_values` [images, max_patches, 768],
padded, with `pixel_attention_mask` [images, max_patches] and `spatial_shapes` [images, 2] = the patch grid (rows,
cols)). Inside the tower the patches of all images are packed into one [patches, width] sequence, so padding costs
nothing. Positions are 2-D rotary (half of each head's dimensions turn with the patch row, half with the column).

The patch embedding is a convolutional stem (ConvStem): separable stride-2 stages from the full-resolution image, so
each patch's vector also sees the pixels around it. Each layer is Budgie's: pre-RMSNorm, attention within the image
with QK-norm, and the deep FFN d -> w1 -> ... -> wn -> d (SiLU, centred dSiLU in the middle), with the text model's
convs made 2-D: Q, K/V and the FFN each read a `conv` x `conv` depthwise conv of their input over the image's patch
grid (not causal; zero at the image's edge), which also carries information across window edges. The 2-D convs run
on the causal-conv1d kernel (depthwise_conv2d). Attention is one of two kinds:

    windowed   every patch attends exactly to the patches of its `window` x `window` square (the grid is cut into
               aligned squares, the last row and column of squares padded); most layers.
    global     every `global_every`-th layer. An image of at most `dense_max_patches` patches gets plain full
               attention. A larger one gets compressed + selected attention, the 2-D, bidirectional form of sparse.py
               (NSA): the grid is cut into `compress` x `compress` blocks, and each block's queries (a query group)
               attend in ONE softmax over
                   compressed  every block of the image pooled into one key and value (sparse.compress_kv: a learned
                               softmax weighting, pool_score [heads, head_dim], 0 = the mean; pooled from the keys
                               before rotary, then rotated at the block's centre);
                   selected    the `select` blocks with the highest compressed attention summed over the group's
                               queries and all heads, every patch of them exactly; blocks of the group's own window
                               are not candidates;
                   local       the patches of the group's `window` x `window` square exactly.
               A screenshot at the 4096-token budget is 16384 patches: ~1024 + 256 + 256 keys per query instead of
               16384.

The pixel shuffle then concatenates each 2 x 2 square of neighbouring patch outputs into one vector, and the
projector LayerNorm -> Linear -> GELU -> Linear maps those to hidden_size. BudgieModel puts the result in place of the
<|image|> placeholders, in order.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .activations import CenteredDSiLU
from .configuration_budgie import BudgieConfig
from .convolution import causal_conv, causal_conv1d_fn
from .norms import QKNorm, RMSNorm
from .sparse import BUDGET, NEG, _attend_groups, compress_kv


def rotary_angles(rows, cols, head_dim, theta):
    """(cos, sin) [n, head_dim] in fp64 for patches at (rows, cols) [n] (any float positions): the first half of
    each head's dimensions rotates with the row, the second with the column, each half-split as RotaryEmbedding."""
    quarter = head_dim // 4
    inv = 1.0 / theta ** (torch.arange(quarter, device=rows.device, dtype=torch.float64) / quarter)
    angles = torch.cat([rows.double()[:, None] * inv, cols.double()[:, None] * inv], dim=-1)
    angles = torch.cat([angles, angles], dim=-1)          # half-split: dimension j and j + head_dim/2 share an angle
    return angles.cos(), angles.sin()


def apply_rotary(x, cos, sin):
    """x [n, heads, head_dim] (or [batch, heads, n, head_dim] with cos/sin broadcast to it) rotated by (cos, sin)."""
    acc = torch.promote_types(x.dtype, torch.float32)
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x.to(acc) * cos.to(acc) + rotated.to(acc) * sin.to(acc)).to(x.dtype)


def grid_squares(h, w, size, fill, offset=0, device=None):
    """The token indices of an h x w row-major grid (offset + r * w + c) cut into aligned size x size squares, the
    grid padded with `fill` to whole squares: [squares, size * size], squares row-major, tokens row-major inside."""
    H, W = -(-h // size) * size, -(-w // size) * size
    grid = torch.full((H, W), fill, dtype=torch.long, device=device)
    grid[:h, :w] = torch.arange(h * w, device=device).view(h, w) + offset
    return grid.view(H // size, size, W // size, size).permute(0, 2, 1, 3).reshape(-1, size * size)


class Layout:
    """Where every packed patch is: its row and column, which image it belongs to, and the windows.

    keep (optional, bool [all patches of all images], image after image, row-major) keeps only those patches in the
    packed sequence (masked image modelling): attention runs among the kept patches only, and the 2-D convs see the
    others as zeros (scattered into the full grid, convolved, gathered back), so a dropped patch never reaches a kept
    one. Windows hold each window's patches first (in grid order) and are cut to the fullest window's count."""

    def __init__(self, spatial_shapes, window, device, keep=None):
        self.shapes = [tuple(int(n) for n in hw) for hw in spatial_shapes.tolist()]
        sizes = [h * w for h, w in self.shapes]
        self.offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        self.n = self.n_dense = sum(sizes)
        self.rows = torch.cat([torch.arange(h, device=device).repeat_interleave(w) for h, w in self.shapes])
        self.cols = torch.cat([torch.arange(w, device=device).repeat(h) for h, w in self.shapes])
        windows = torch.cat([grid_squares(h, w, window, self.n, off, device)
                             for (h, w), off in zip(self.shapes, self.offsets)])        # [windows, window^2], pad = n
        self.keep_idx = None
        counts = sizes
        if keep is not None:
            keep = keep.to(device=device, dtype=torch.bool)
            if keep.shape != (self.n_dense,):
                raise ValueError(f"keep must be [{self.n_dense}] (every patch of every image), got {tuple(keep.shape)}")
            self.keep_idx = keep.nonzero().squeeze(1)
            self.n = len(self.keep_idx)
            remap = torch.full((self.n_dense + 1,), self.n, dtype=torch.long, device=device)
            remap[self.keep_idx] = torch.arange(self.n, device=device)
            windows = remap[windows]
            self.rows, self.cols = self.rows[self.keep_idx], self.cols[self.keep_idx]
            counts = [int(keep[off:off + s].sum()) for off, s in zip(self.offsets, sizes)]
        windows = windows.sort(-1).values                                   # real patches first, padding (= n) last
        windows = windows[:, :max(1, int((windows < self.n).sum(-1).max()))] if len(windows) else windows
        self.windows = windows
        self.window_ok = self.windows < self.n
        # each image's (h, w, first packed patch, packed patches); the same as shapes / offsets without keep
        self.spans = [(h, w, sum(counts[:i]), c) for i, ((h, w), c) in enumerate(zip(self.shapes, counts))]
        # runs of consecutive images of the same shape: (h, w, first image, images, first patch); each run's patches
        # are one contiguous [images, h, w, C] block, so convs and the stem take a run in one call
        self.runs = []
        for i, (hw, off) in enumerate(zip(self.shapes, self.offsets)):
            if self.runs and self.runs[-1][:2] == hw:
                h, w, first, count, start = self.runs[-1]
                self.runs[-1] = (h, w, first, count + 1, start)
            else:
                self.runs.append((*hw, i, 1, off))


def depthwise_conv2d(x, weight, rows=None):
    """x [n, H, W, C] (channels last), weight [C, k, k] (k odd) -> [n, H, W, C]: out[c, r, s] = sum_ij weight[c, i, j]
    * x[c, r + i - p, s + j - p], p = k // 2, zero outside each image.

    On a GPU with the causal-conv1d kernel (k <= 4) it is built from 1-D convs that run on that kernel: every image
    row is one sequence, the k kernel rows are k stacked copies of the channels in ONE call, each made centred by
    padding p zeros on the right and dropping the first p outputs, and the k results are summed with kernel row i
    shifted by i - p rows. Elsewhere (the CPU) it is one oneDNN depthwise conv2d. rows=True/False forces a path."""
    n, H, W, C = x.shape
    k = weight.shape[-1]
    p = k // 2
    weight = weight.to(x.dtype)
    if rows is None:
        rows = x.is_cuda and causal_conv1d_fn is not None and k <= 4
    if not rows:
        return F.conv2d(x.permute(0, 3, 1, 2), weight[:, None], padding=p, groups=C).permute(0, 2, 3, 1)
    seq = F.pad(x.reshape(n * H, W, C), (0, 0, 0, p)).repeat(1, 1, k)           # [n*H, W + p, k*C], channels last
    y = causal_conv(seq.transpose(1, 2), weight.transpose(0, 1).reshape(k * C, k))[..., p:]   # [n*H, k*C, W]
    y = y.transpose(1, 2).reshape(n, H, W, k, C)
    out = y[:, :, :, p]
    for i in range(k):
        shift = i - p                                                              # out[r] += y_i[r + shift]
        if shift > 0:
            out = out + F.pad(y[:, shift:, :, i], (0, 0, 0, 0, 0, shift))
        elif shift < 0:
            out = out + F.pad(y[:, :shift, :, i], (0, 0, 0, 0, -shift, 0))
    return out


class DepthwiseConv2d(nn.Module):
    """A k x k depthwise conv over each image's patch grid (zero padding at the image's edge, so images never mix),
    one filter per channel, no bias. Starts as the identity (only the centre tap is 1), as Budgie's CausalConv does."""

    def __init__(self, channels, k):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.zeros(channels, k, k))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weight.zero_()
        self.weight[:, self.k // 2, self.k // 2] = 1.0

    def forward(self, x):
        """x [images, H, W, C] (channels last)."""
        return depthwise_conv2d(x, self.weight)

    def packed(self, x, layout):
        """x [patches of all images, C] (Layout order) -> the same, each image convolved over its own grid; one call
        per run of same-shape images."""
        C = x.shape[-1]
        if layout.keep_idx is not None:                 # dropped patches are zeros in the grid
            x = x.new_zeros(layout.n_dense, C).index_copy(0, layout.keep_idx, x)
        outs = [self(x[start:start + n * h * w].view(n, h, w, C)).reshape(n * h * w, C)
                for h, w, _, n, start in layout.runs]
        out = outs[0] if len(outs) == 1 else torch.cat(outs)
        return out if layout.keep_idx is None else out[layout.keep_idx]


def patches_to_image(patches, h, w, patch, channels):
    """[images, h*w, patch*patch*channels] row-major patches, each (y, x, channel) flattened (processing_budgie's
    layout) -> the images [images, h*patch, w*patch, channels] (channels last)."""
    n = patches.shape[0]
    return (patches.reshape(n, h, w, patch, patch, channels).permute(0, 1, 3, 2, 4, 5)
            .reshape(n, h * patch, w * patch, channels))


class ConvStem(nn.Module):
    """The patch embedding as a convolutional stem (Xiao et al. 2021, "Early convolutions help transformers see
    better"), in separable form: log2(patch_size) stages, each a k x k depthwise conv over the full-resolution image
    (DepthwiseConv2d), a 2 x 2 space-to-depth (halving the resolution, 4x the channels) and a Linear to the stage's
    width, then RMSNorm + SiLU on every stage but the last. Widths: num_channels -> *stem -> hidden_size. Unlike a
    Linear patch embedding, a patch's vector sees the pixels around the patch too."""

    def __init__(self, t):
        super().__init__()
        widths = [t["num_channels"], *t["stem"], t["hidden_size"]]
        self.patch, self.channels = t["patch_size"], t["num_channels"]
        k = t["conv"]
        self.convs = nn.ModuleList([DepthwiseConv2d(c, k) for c in widths[:-1]]) if k else None
        self.proj = nn.ModuleList([nn.Linear(4 * a, b) for a, b in zip(widths, widths[1:])])
        self.norms = nn.ModuleList([RMSNorm(b, t["rms_norm_eps"]) for b in widths[1:-1]])

    def images(self, img):
        """img [images, H, W, channels] -> [images * H / patch * W / patch, hidden_size], each image row-major."""
        x = img
        for s, proj in enumerate(self.proj):
            if self.convs is not None:
                x = self.convs[s](x)
            n, H, W, C = x.shape
            x = x.reshape(n, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(n, H // 2, W // 2, 4 * C)  # (dy, dx, C)
            x = proj(x)
            if s < len(self.norms):
                x = F.silu(self.norms[s](x))
        return x.reshape(-1, x.shape[-1])

    def forward(self, pixel_values, layout):
        dtype = self.proj[0].weight.dtype
        outs = [self.images(patches_to_image(pixel_values[first:first + n, :h * w].to(dtype), h, w, self.patch, self.channels))
                for h, w, first, n, _ in layout.runs]
        return outs[0] if len(outs) == 1 else torch.cat(outs)


def _pad_row(x):
    return torch.cat([x, x.new_zeros(1, *x.shape[1:])])


def window_attention(q, k, v, layout):
    """q, k, v [n, heads, D] packed; every patch attends to the patches of its window exactly."""
    idx, ok = layout.windows, layout.window_ok
    qw, kw, vw = (_pad_row(x)[idx].transpose(1, 2) for x in (q, k, v))         # [windows, heads, window^2, D]
    out = F.scaled_dot_product_attention(qw, kw, vw, attn_mask=ok[:, None, None, :]).transpose(1, 2)
    return q.new_zeros(q.shape).index_copy(0, idx[ok], out[ok])


def sparse_attention_2d(q, k, v, k_raw, h, w, spec, pool_score):
    """Compressed + selected attention over one image of h x w patches (row-major). q, k [T, heads, D] after
    rotary, k_raw the keys before it, v [T, heads, D]. Returns [T, heads, D]."""
    m, W, n, theta = spec["compress"], spec["window"], spec["select"], spec["rope_theta"]
    T, H, D = q.shape
    scale = D**-0.5
    dev, cd = q.device, q.dtype
    blk = grid_squares(h, w, m, T, device=dev)                                 # [nb, m*m] tokens of each block
    tok_ok = blk < T
    blk_ok = tok_ok.any(-1)                                                    # [nb]
    nb, g = blk.shape
    qp, kp, vp, krp = (_pad_row(x).permute(1, 0, 2)[None] for x in (q, k, v, k_raw))   # [1, heads, T + 1, D]

    # compressed keys / values: pooled per block (block-major), the keys rotated at the block's centre
    flat = blk.reshape(-1)
    kc, vc, _ = compress_kv(krp[:, :, flat], vp[:, :, flat], pool_score, g, key_ok=tok_ok.reshape(1, -1))
    nbc = -(-w // m)                                                           # blocks per row (grid padded to m)
    centre = (m - 1) / 2
    br = torch.arange(nb, device=dev) // nbc
    bc = torch.arange(nb, device=dev) % nbc
    cos, sin = rotary_angles(br * m + centre, bc * m + centre, D, theta)
    kc = apply_rotary(kc, cos, sin)                                            # [1, heads, nb, D]

    # each block's window: blocks per window side = W // m
    per = W // m
    win = (br // per) * -(-w // W) + bc // per                                 # [nb] window id of each block
    win_tok = grid_squares(h, w, W, T, device=dev)                             # [windows, W*W]
    locc = win_tok[win]                                                        # [nb, W*W] local keys of each group
    loc_ok = (locc < T)[None, :, None, :]                                      # [1, nG, 1, Lw]

    qg = qp[:, :, blk][:, :, None]                                             # [1, heads, 1, nb, g, D]
    # selection: the group's summed softmax over the compressed blocks (no gradient, as in sparse.py)
    with torch.no_grad():
        S = torch.zeros(nb, nb, device=dev, dtype=kc.dtype)
        step = max(1, BUDGET // max(1, H * g * nb))
        kcq = kc[0].to(cd)
        for a in range(0, nb, step):
            z = min(nb, a + step)
            lg = torch.einsum("hngd,hjd->hngj", qg[0, :, 0, a:z], kcq).to(S.dtype) * scale
            p = torch.softmax(lg.masked_fill(~blk_ok, NEG), -1) * tok_ok[a:z, :, None]
            S[a:z] = p.sum((0, 2))
    nsel = min(n, nb)
    cand = blk_ok[None, :] & (win[None, :] != win[:, None])                    # not the group's own window
    top, idx = S.masked_fill(~cand, NEG).topk(nsel, -1)                       # [nb, nsel]
    sel = blk[idx].reshape(nb, nsel * g)[None]                                 # [1, nG, L]
    sel_ok = ((top > NEG)[..., None] & tok_ok[idx]).reshape(nb, nsel * g)[None]

    c_mask = blk_ok[None, None, :].expand(1, nb, nb)
    L, Lw = sel.shape[-1], locc.shape[-1]
    per_group = H * g * (nb + L + Lw) + 2 * H * (L + Lw) * D
    step = max(1, BUDGET // per_group)
    outs = []
    for a in range(0, nb, step):
        z = min(nb, a + step)
        args = (qg[:, :, :, a:z], kp, vp, kc, vc, c_mask[:, a:z], sel[:, a:z], sel_ok[:, a:z], locc[a:z],
                loc_ok[:, a:z], None, scale)
        if torch.is_grad_enabled() and nb > step:
            outs.append(checkpoint(_attend_groups, *args, use_reentrant=False))
        else:
            outs.append(_attend_groups(*args))
    out = torch.cat(outs, 3)[0, :, 0].permute(1, 2, 0, 3).reshape(nb * g, H, D)   # [tokens of blk order, heads, D]
    ok = tok_ok.reshape(-1)
    return q.new_zeros(T, H, D).index_copy(0, flat[ok], out[ok].to(cd))


def global_attention(q, k, v, k_raw, layout, spec, pool_score):
    """The global layers: full attention up to dense_max_patches patches, compressed + selected above. Consecutive
    dense images with the same number of patches go through one batched SDPA; the packed sequence is cut with one
    split (per-image slices cost a full-size gradient buffer each in the backward pass)."""
    groups = []                                         # [first patch, images, patches each, dense, (h, w)]
    for h, w, off, count in layout.spans:
        dense = h * w <= spec["dense_max_patches"]
        if layout.keep_idx is not None and not dense:
            raise NotImplementedError(f"keep (masked patches) with a {h} x {w} image: compressed + selected attention "
                                      f"needs the full grid; masked images must be <= dense_max_patches patches")
        if groups and dense and groups[-1][3] and groups[-1][2] == count:
            groups[-1][1] += 1
        else:
            groups.append([off, 1, count, dense, (h, w)])
    sizes = [n * c for _, n, c, _, _ in groups]
    T, H, D = q.shape
    outs = []
    for (_, n, c, dense, (h, w)), qi, ki, vi, kri in zip(groups, *(x.split(sizes) for x in (q, k, v, k_raw))):
        if c == 0:
            outs.append(qi)
        elif dense:
            qb, kb, vb = (x.view(n, c, H, D).transpose(1, 2) for x in (qi, ki, vi))
            outs.append(F.scaled_dot_product_attention(qb, kb, vb).transpose(1, 2).reshape(n * c, H, D))
        else:
            outs.append(sparse_attention_2d(qi, ki, vi, kri, h, w, spec, pool_score))
    return outs[0] if len(outs) == 1 else torch.cat(outs)


class VisionAttention(nn.Module):
    def __init__(self, t, is_global):
        super().__init__()
        self.t, self.is_global = t, is_global
        self.heads, self.head_dim = t["num_attention_heads"], t["head_dim"]
        d, inner = t["hidden_size"], t["num_attention_heads"] * t["head_dim"]
        # as in the text model, Q reads a depthwise conv of the input and K/V read a second one (here 2-D, k x k)
        self.q_conv = DepthwiseConv2d(d, t["conv"]) if t["conv"] else None
        self.kv_conv = DepthwiseConv2d(d, t["conv"]) if t["conv"] else None
        self.q_proj = nn.Linear(d, inner, bias=False)
        self.kv_proj = nn.Linear(d, 2 * inner, bias=False)
        self.o_proj = nn.Linear(inner, d, bias=False)
        self.q_norm = QKNorm(self.head_dim, t["rms_norm_eps"])
        self.k_norm = QKNorm(self.head_dim, t["rms_norm_eps"])
        # weights of the keys pooled into a compressed block (global layers only); zero = the plain mean
        self.pool_score = nn.Parameter(torch.zeros(self.heads, self.head_dim)) if is_global else None

    def project(self, xq, xkv, cos, sin):
        """The per-patch part (no layout): projections, QK-norm, rotary. BudgieViT.compile_layers compiles it."""
        n = xq.shape[0]
        q = self.q_proj(xq).view(n, self.heads, self.head_dim)
        k, v = self.kv_proj(xkv).view(n, 2, self.heads, self.head_dim).unbind(1)
        q, k_raw = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary(q, cos[:, None], sin[:, None]), apply_rotary(k_raw, cos[:, None], sin[:, None])
        return q, k, v, k_raw

    def forward(self, x, cos, sin, layout):
        """-> the attention output before o_proj, [patches, heads * head_dim]."""
        xq = self.q_conv.packed(x, layout) if self.q_conv is not None else x
        xkv = self.kv_conv.packed(x, layout) if self.kv_conv is not None else x
        q, k, v, k_raw = self.project(xq, xkv, cos, sin)
        if self.is_global:
            out = global_attention(q, k, v, k_raw, layout, self.t, self.pool_score)
        else:
            out = window_attention(q, k, v, layout)
        return out.reshape(x.shape[0], -1)


class VisionFeedForward(nn.Module):
    """Budgie's deep FFN with its conv made 2-D (k x k depthwise over the image, not causal): conv, then d -> w1 ->
    ... -> wn -> d, SiLU after every hidden layer except the middle one (n // 2), which gets the centred dSiLU."""

    def __init__(self, t):
        super().__init__()
        self.conv = DepthwiseConv2d(t["hidden_size"], t["conv"]) if t["conv"] else None
        widths = [t["hidden_size"], *t["intermediate_size"], t["hidden_size"]]
        self.layers = nn.ModuleList([nn.Linear(widths[i], widths[i + 1], bias=False) for i in range(len(widths) - 1)])
        hidden = len(widths) - 2
        self.activations = nn.ModuleList([CenteredDSiLU() if i == hidden // 2 else nn.SiLU() for i in range(hidden)])

    def core(self, x):
        """The linear layers and activations (no layout). BudgieViT.compile_layers compiles it."""
        for linear, activation in zip(self.layers[:-1], self.activations):
            x = activation(linear(x))
        return self.layers[-1](x)

    def forward(self, x, layout):
        return self.core(self.conv.packed(x, layout) if self.conv is not None else x)


class VisionLayer(nn.Module):
    def __init__(self, t, is_global):
        super().__init__()
        self.attn_norm = RMSNorm(t["hidden_size"], t["rms_norm_eps"])
        self.attn = VisionAttention(t, is_global)
        self.ffn_norm = RMSNorm(t["hidden_size"], t["rms_norm_eps"])
        self.ffn = VisionFeedForward(t)

    def pre(self, x):
        return self.attn_norm(x)

    def mid(self, x, a):
        """x + o_proj(attention), and that normed for the FFN."""
        x = x + self.attn.o_proj(a)
        return x, self.ffn_norm(x)

    def forward(self, x, cos, sin, layout):
        x, h = self.mid(x, self.attn(self.pre(x), cos, sin, layout))
        return x + self.ffn(h, layout)


class BudgieViT(nn.Module):
    """The vision tower: the patch embedding (a ConvStem, or a Linear when `stem` is None), num_hidden_layers
    VisionLayers (every global_every-th one global, the rest windowed), a final RMSNorm. Works on the packed patches
    of all images."""

    def __init__(self, t):
        super().__init__()
        self.t = t
        self.stem = ConvStem(t) if t["stem"] is not None else None
        self.patch_embed = None if self.stem is not None else nn.Linear(t["num_channels"] * t["patch_size"] ** 2, t["hidden_size"])
        self.layers = nn.ModuleList([VisionLayer(t, (i + 1) % t["global_every"] == 0) for i in range(t["num_hidden_layers"])])
        self.norm = RMSNorm(t["hidden_size"], t["rms_norm_eps"])
        self.gradient_checkpointing = False   # set by a trainer: recompute each layer in the backward pass

    def compile_layers(self, **kwargs):
        """torch.compile the per-patch parts of every layer (norms, projections, QK-norm, rotary, the FFN's linear
        layers and activations, residual adds), with dynamic shapes so any number of patches reuses one graph. The
        layout-dependent parts (convs, window gathers, global attention) stay eager: their shapes change with every
        image size. All layers share the compiled code (their parameters are graph inputs)."""
        kwargs.setdefault("dynamic", True)
        for layer in self.layers:
            layer.pre = torch.compile(layer.pre, **kwargs)
            layer.mid = torch.compile(layer.mid, **kwargs)
            layer.attn.project = torch.compile(layer.attn.project, **kwargs)
            layer.ffn.core = torch.compile(layer.ffn.core, **kwargs)
        return self

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes, keep=None):
        """-> [patches of all images, hidden_size], image after image, each row-major. With keep (bool [all patches],
        Layout order) only the kept patches, in order: the dropped ones' pixels are zeroed before the stem and they
        are left out of every layer (masked image modelling; images of at most dense_max_patches patches)."""
        layout = Layout(spatial_shapes, self.t["window"], pixel_values.device, keep)
        if keep is not None:                            # dropped patches' pixels never reach the stem
            valid = pixel_attention_mask.bool()
            per_image = torch.zeros(valid.shape, dtype=pixel_values.dtype, device=pixel_values.device)
            per_image[valid] = keep.to(device=pixel_values.device, dtype=pixel_values.dtype)
            pixel_values = pixel_values * per_image[..., None]
        if self.stem is not None:
            if self.gradient_checkpointing and torch.is_grad_enabled():   # full-resolution activations: recompute
                x = checkpoint(self.stem, pixel_values, layout, use_reentrant=False)
            else:
                x = self.stem(pixel_values, layout)
        else:
            x = self.patch_embed(pixel_values[pixel_attention_mask.bool()].to(self.patch_embed.weight.dtype))
        if layout.keep_idx is not None:
            x = x[layout.keep_idx]
        acc = torch.promote_types(x.dtype, torch.float32)
        cos, sin = (c.to(acc) for c in rotary_angles(layout.rows, layout.cols, self.t["head_dim"], self.t["rope_theta"]))
        for layer in self.layers:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(layer, x, cos, sin, layout, use_reentrant=False)
            else:
                x = layer(x, cos, sin, layout)
        return self.norm(x)


class BudgieVision(nn.Module):
    def __init__(self, config: BudgieConfig):
        super().__init__()
        v = config.vision_config
        self.shuffle = v["pixel_shuffle"]
        self.tower = BudgieViT(v["tower"])
        width = v["tower"]["hidden_size"] * self.shuffle**2
        self.proj_norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):
        """-> [sum over images of rows * cols / pixel_shuffle^2, hidden_size]: every image's tokens, row-major, image
        after image."""
        s = self.shuffle
        shapes = [tuple(int(n) for n in hw) for hw in spatial_shapes.tolist()]
        for (h, w), valid in zip(shapes, pixel_attention_mask.sum(1).tolist()):
            if h % s or w % s or h * w != valid:
                raise ValueError(f"image patch grid {h} x {w} must be a multiple of pixel_shuffle {s} each way and match its "
                                 f"{valid} unpadded patches (BudgieProcessor sizes images this way)")
        out = self.tower(pixel_values, pixel_attention_mask, spatial_shapes)
        c, merged, off = out.shape[-1], [], 0
        for h, w in shapes:
            img = out[off:off + h * w]
            off += h * w
            merged.append(img.view(h // s, s, w // s, s, c).permute(0, 2, 1, 3, 4).reshape(-1, s * s * c))
        h = torch.cat(merged).to(self.fc1.weight.dtype)
        return self.fc2(F.gelu(self.fc1(self.proj_norm(h))))
