"""BudgieViT with `keep` (masked image modelling, train_tower_mae.py): only the kept patches go through the tower."""

import pytest
import torch

from budgie import DEFAULT_VISION_CONFIG
from budgie.vision import BudgieViT, Layout, apply_rotary, rotary_angles, window_attention

# 8 px patches; layer 0 windowed (4 x 4), layer 1 global (dense up to 64 patches); 3 x 3 convs, a 3-stage conv stem
TOWER = {**DEFAULT_VISION_CONFIG["tower"], "hidden_size": 16, "intermediate_size": [24, 32, 24], "num_hidden_layers": 2,
         "num_attention_heads": 2, "head_dim": 8, "patch_size": 8, "window": 4, "global_every": 2, "compress": 2,
         "select": 2, "dense_max_patches": 64, "conv": 3, "stem": [4, 8]}
PATCH_DIM = 8 * 8 * 3


def tower(seed=0, **over):
    torch.manual_seed(seed)
    t = BudgieViT({**TOWER, **over}).double().eval()
    with torch.no_grad():                               # convs off identity, so they mix neighbours
        for m in t.modules():
            if hasattr(m, "k") and hasattr(m, "weight") and m.weight.dim() == 3:
                m.weight.normal_(0, 0.3)
    return t


def batch(shapes, seed=1):
    g = torch.Generator().manual_seed(seed)
    longest = max(h * w for h, w in shapes)
    px = torch.zeros(len(shapes), longest, PATCH_DIM, dtype=torch.float64)
    mask = torch.zeros(len(shapes), longest, dtype=torch.int32)
    for i, (h, w) in enumerate(shapes):
        px[i, :h * w] = torch.rand(h * w, PATCH_DIM, generator=g, dtype=torch.float64) * 2 - 1
        mask[i, :h * w] = 1
    return px, mask, torch.tensor(shapes)


def random_keep(shapes, ratio=0.25, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(sum(h * w for h, w in shapes), generator=g) < ratio


SHAPES = [(6, 6), (4, 8), (6, 6), (2, 4)]


def test_keeping_every_patch_changes_nothing():
    t = tower()
    px, mask, shapes = batch(SHAPES)
    with torch.no_grad():
        full = t(px, mask, shapes)
        kept = t(px, mask, shapes, keep=torch.ones(len(full), dtype=torch.bool))
    torch.testing.assert_close(kept, full, rtol=0, atol=1e-12)


def test_kept_patches_never_see_dropped_pixels():
    t = tower()
    px, mask, shapes = batch(SHAPES)
    keep = random_keep(SHAPES)
    with torch.no_grad():
        a = t(px, mask, shapes, keep=keep)
        other = px.clone()
        valid = mask.bool()
        flat = other[valid]
        flat[~keep] = torch.rand_like(flat[~keep]) * 2 - 1       # new pixels in every dropped patch
        other[valid] = flat
        b = t(other, mask, shapes, keep=keep)
        flat2 = px[valid].clone()
        flat2[keep.nonzero()[0, 0]] += 0.5                         # and a kept patch does matter
        moved = px.clone()
        moved[valid] = flat2
        c = t(moved, mask, shapes, keep=keep)
    assert a.shape == (int(keep.sum()), TOWER["hidden_size"])
    torch.testing.assert_close(b, a, rtol=0, atol=1e-12)
    assert (c - a).abs().max() > 1e-6


@pytest.mark.parametrize("stem", [None, [4, 8]])
def test_masked_images_never_mix_in_a_batch(stem):
    t = tower(stem=stem)
    px, mask, shapes = batch(SHAPES)
    keep = random_keep(SHAPES, ratio=0.4, seed=5)
    with torch.no_grad():
        got = t(px, mask, shapes, keep=keep)
        outs, off = [], 0
        for i, (h, w) in enumerate(SHAPES):
            k_img = keep[off:off + h * w]
            off += h * w
            if k_img.any():
                n = h * w
                outs.append(t(px[i:i + 1, :n], mask[i:i + 1, :n], shapes[i:i + 1], keep=k_img))
    torch.testing.assert_close(got, torch.cat(outs), rtol=0, atol=1e-12)


def test_window_attention_with_keep_is_full_attention_among_each_windows_kept_patches():
    shapes = torch.tensor([(8, 8), (5, 7)])
    keep = random_keep([(8, 8), (5, 7)], ratio=0.3, seed=7)
    lay = Layout(shapes, 4, "cpu", keep)
    n, H, D = lay.n, 2, 8
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(n, H, D, generator=g, dtype=torch.float64) for _ in range(3))
    got = window_attention(q, k, v, lay)
    dense = Layout(shapes, 4, "cpu")
    pos = torch.full((dense.n,), -1, dtype=torch.long)
    pos[keep] = torch.arange(n)
    ref = torch.zeros_like(q)
    for row in dense.windows:
        members = pos[row[row < dense.n]]
        members = members[members >= 0]
        if len(members):
            a = torch.softmax(torch.einsum("ihd,jhd->hij", q[members], k[members]) * D ** -0.5, -1)
            ref[members] = torch.einsum("hij,jhd->ihd", a, v[members])
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-12)
    assert lay.windows.shape[1] < 16                            # windows cut to the fullest window's kept count


def test_rotary_positions_follow_the_kept_patches():
    shapes = torch.tensor([(3, 5)])
    keep = torch.tensor([i % 3 == 0 for i in range(15)])
    lay = Layout(shapes, 4, "cpu", keep)
    assert lay.rows.tolist() == [i // 5 for i in range(15) if i % 3 == 0]
    assert lay.cols.tolist() == [i % 5 for i in range(15) if i % 3 == 0]


def test_masked_image_above_dense_max_patches_is_refused():
    t = tower(dense_max_patches=16)
    px, mask, shapes = batch([(6, 6)])
    with pytest.raises(NotImplementedError):
        t(px, mask, shapes, keep=random_keep([(6, 6)]))


def test_masked_tower_trains():
    t = tower().float().train()
    px, mask, shapes = batch(SHAPES)
    keep = random_keep(SHAPES)
    t(px.float(), mask, shapes, keep=keep).square().mean().backward()
    assert all(p.grad is not None for n, p in t.named_parameters() if "pool_score" not in n)
