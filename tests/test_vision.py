"""Image input (vision.py, processing_budgie.py) on a tiny model with a tiny random vision tower, on the CPU."""

import copy
import pathlib
import random

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from budgie import DEFAULT_VISION_CONFIG, BudgieConfig, BudgieForCausalLM, BudgieProcessor
from budgie import vision
from budgie.vision import DepthwiseConv2d, Layout, depthwise_conv2d, patches_to_image, apply_rotary, global_attention, rotary_angles, sparse_attention_2d, window_attention
from budgie.processing_budgie import (IMAGE, IMAGE_END, IMAGE_ROW, IMAGE_START, collapse_images, expand_images, fit_image,
                                      image_patches, image_span)

from test_model import VOCAB, tiny_config, tiny_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMG, START, END, ROW = 18, 19, 20, 21
# 8 px patches x 2 x 2 shuffle = 16 px per token. Layer 0 windowed (4 x 4 patches), layer 1 global and always
# compressed + selected (2 x 2 blocks, top 2), so the model tests run both kinds of attention; 3 x 3 convs and a
# 3-stage conv stem.
TINY_VISION = {**DEFAULT_VISION_CONFIG,
               "tower": {**DEFAULT_VISION_CONFIG["tower"], "hidden_size": 16, "intermediate_size": [24, 32, 24],
                         "num_hidden_layers": 2, "num_attention_heads": 2, "head_dim": 8, "patch_size": 8,
                         "window": 4, "global_every": 2, "compress": 2, "select": 2, "dense_max_patches": 0,
                         "conv": 3, "stem": [4, 8]},
               "min_image_tokens": 4, "max_image_tokens": 48}
PATCH_DIM = 8 * 8 * 3


def vision_model(seed=0):
    model = tiny_model(seed=seed)
    torch.manual_seed(seed + 1)
    model.add_vision(TINY_VISION)
    return model.eval()


def span_ids(rows, cols):
    return [START, *([IMG] * cols + [ROW]) * rows, END]


def image_ids(prefix, grids, suffix, seed=3):
    g = torch.Generator().manual_seed(seed)
    ids = [1, *torch.randint(30, VOCAB, (prefix,), generator=g).tolist()]
    for r, c in grids:
        ids += [*span_ids(r, c), *torch.randint(30, VOCAB, (2,), generator=g).tolist()]
    ids += torch.randint(30, VOCAB, (suffix,), generator=g).tolist()
    return torch.tensor([ids])


def pixels(grids, seed=4, longest=None):
    """Random patches for images of grids (token rows, cols) in the processor's padded layout."""
    g = torch.Generator().manual_seed(seed)
    n = [4 * r * c for r, c in grids]
    longest = longest or max(n)
    px, mask = torch.zeros(len(grids), longest, PATCH_DIM), torch.zeros(len(grids), longest, dtype=torch.int32)
    for i, k in enumerate(n):
        px[i, :k], mask[i, :k] = torch.randn(k, PATCH_DIM, generator=g), 1
    return {"pixel_values": px, "pixel_attention_mask": mask, "spatial_shapes": torch.tensor([[2 * r, 2 * c] for r, c in grids])}


def test_default_text_plus_vision_is_under_2b():
    cfg = BudgieConfig(vision_config=DEFAULT_VISION_CONFIG)
    t, d = cfg.vision_config["tower"], cfg.hidden_size
    widths = [t["hidden_size"], *t["intermediate_size"], t["hidden_size"]]
    k2 = t["conv"] ** 2
    layer = (4 * t["num_attention_heads"] * t["head_dim"] * t["hidden_size"] + 2 * t["head_dim"]      # attention + QK-norms
             + sum(a * b for a, b in zip(widths, widths[1:])) + 2 * t["hidden_size"]                # FFN + 2 RMSNorms
             + 3 * k2 * t["hidden_size"])                                                         # Q, K/V, FFN convs
    stem_w = [t["num_channels"], *t["stem"], t["hidden_size"]]
    stem = (sum(k2 * c for c in stem_w[:-1]) + sum((4 * a + 1) * b for a, b in zip(stem_w, stem_w[1:]))
            + sum(stem_w[1:-1]))                                                                  # convs, Linears, norms
    n_global = t["num_hidden_layers"] // t["global_every"]
    tower = stem + t["num_hidden_layers"] * layer + t["hidden_size"] + n_global * t["num_attention_heads"] * t["head_dim"]
    wide = 4 * t["hidden_size"]
    projector = 2 * wide + wide * d + d + d * d + d
    with torch.device("meta"):
        text = sum(p.numel() for p in BudgieForCausalLM(BudgieConfig()).parameters())
        full = BudgieForCausalLM(cfg)
    assert sum(p.numel() for p in full.model.vision.tower.stem.parameters()) == stem
    assert sum(p.numel() for p in full.model.vision.tower.parameters()) == tower
    assert sum(p.numel() for p in full.parameters()) - text == tower + projector
    assert text + tower + projector < 2_000_000_000
    assert cfg.image_pixels_per_token == 32


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("rows", [False, True])                  # oneDNN conv2d / the causal-conv1d row decomposition
def test_depthwise_conv2d_is_a_2d_conv_with_zero_padding(k, rows):
    x = torch.randn(2, 7, 9, 5, dtype=torch.float64)                  # [images, H, W, C]
    w = torch.randn(5, k, k, dtype=torch.float64)
    ref = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w[:, None], padding=k // 2, groups=5).permute(0, 2, 3, 1)
    assert torch.allclose(depthwise_conv2d(x, w, rows=rows), ref, atol=1e-12)
    assert torch.equal(DepthwiseConv2d(5, k).double()(x), x)                   # starts as the identity


def test_compiled_layers_match_eager():
    torch.manual_seed(0)
    model = vision_model()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("conv.weight") and p.dim() == 3:
                p.normal_()
    eager = model.model.vision.tower
    compiled = copy.deepcopy(eager).compile_layers()
    grids = [(2, 3), (3, 2)]
    px = pixels(grids)
    a, b = eager(**px), compiled(**px)
    assert torch.allclose(a, b, atol=1e-4)
    ga = torch.autograd.grad(a.pow(2).sum(), list(eager.parameters()))
    gb = torch.autograd.grad(b.pow(2).sum(), list(compiled.parameters()))
    assert all(torch.allclose(x, y, atol=1e-3, rtol=1e-3) for x, y in zip(ga, gb))


def test_patches_to_image_inverts_the_processor():
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 70, 3), dtype=np.uint8))
    rows, cols = 3, 5
    patches = image_patches(img, rows, cols, patch=8, shuffle=2)
    expect = torch.from_numpy(np.asarray(img.convert("RGB").resize((cols * 16, rows * 16), Image.Resampling.BILINEAR),
                                         dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1
    assert torch.equal(patches_to_image(patches[None], rows * 2, cols * 2, 8, 3)[0], expect.permute(1, 2, 0))


def test_conv_stem_sees_past_its_patch_only_once_its_convs_are_trained():
    model = vision_model()
    stem = model.model.vision.tower.stem
    layout = Layout(torch.tensor([[4, 6]]), 4, "cpu")
    px = pixels([(2, 3)])["pixel_values"]
    px2 = px.clone()
    px2[0, 0, :] += 1.0                                            # every pixel of patch (0, 0)
    with torch.no_grad():
        moved = (stem(px2, layout) - stem(px, layout)).abs().sum(-1) > 0
        assert moved.tolist() == [True] + [False] * 23             # identity convs: a Linear patch embedding in effect
        for conv in stem.convs:
            conv.weight.normal_()
        moved = (stem(px2, layout) - stem(px, layout)).abs().sum(-1) > 0
    assert moved[0] and moved[1] and moved[6] and not moved[23]   # its right and lower neighbours, not the far corner


def test_layer_convs_change_each_image_only_from_its_own_patches():
    model = vision_model()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("conv.weight") and p.dim() == 3:
                p.normal_()                                        # every 2-D conv non-trivial
    tower = model.model.vision.tower
    grids = [(2, 3), (1, 2)]
    px = pixels(grids)
    with torch.no_grad():
        both = tower(**px)
        alone = tower(px["pixel_values"][1:, :8], px["pixel_attention_mask"][1:, :8], px["spatial_shapes"][1:])
    assert torch.allclose(both[24:], alone, atol=1e-5)


def test_rotary_2d_depends_only_on_the_row_and_column_offset():
    cos, sin = rotary_angles(torch.arange(12) // 4, torch.arange(12) % 4, 8, 10000.0)   # a row-major 3 x 4 grid
    q, k = torch.randn(1, 8, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64)
    qr = apply_rotary(q.expand(12, 8), cos, sin)
    kr = apply_rotary(k.expand(12, 8), cos, sin)
    score = qr @ kr.T                                              # [12 queries, 12 keys]
    pos = [(i // 4, i % 4) for i in range(12)]
    by_offset = {}
    for i, (r1, c1) in enumerate(pos):
        for j, (r2, c2) in enumerate(pos):
            by_offset.setdefault((r2 - r1, c2 - c1), []).append(float(score[i, j]))
    assert all(max(v) - min(v) < 1e-12 for v in by_offset.values())
    assert len({round(v[0], 6) for v in by_offset.values()}) > 10   # and different offsets do differ


# --- the attention kinds, against references written from their definitions (fp64) ---

SPEC = {"compress": 2, "window": 4, "select": 3, "rope_theta": 10000.0, "dense_max_patches": 0}


def qkv(T, H=2, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k_raw, v = (torch.randn(T, H, D, generator=g, dtype=torch.float64) for _ in range(3))
    return q, k_raw, v, torch.randn(H, D, generator=g, dtype=torch.float64) * 0.7


def rotate(x, h, w, theta=10000.0):
    r = torch.arange(h * w)
    cos, sin = rotary_angles(r // w, r % w, x.shape[-1], theta)
    return apply_rotary(x, cos[:, None], sin[:, None])


def reference_sparse_2d(q, k, v, k_raw, h, w, spec, pool_score):
    """Compressed + selected attention, query by query, from the definition in vision.py's docstring."""
    m, W, n = spec["compress"], spec["window"], spec["select"]
    T, H, D = q.shape
    scale = D**-0.5
    pos = [(i // w, i % w) for i in range(T)]
    blocks = {}
    for i, (r, c) in enumerate(pos):
        blocks.setdefault((r // m, c // m), []).append(i)
    bkeys = sorted(blocks)
    window_of = lambda b: (b[0] * m // W, b[1] * m // W)
    kc, vc = torch.zeros(H, len(bkeys), D, dtype=q.dtype), torch.zeros(H, len(bkeys), D, dtype=q.dtype)
    for j, b in enumerate(bkeys):
        toks = blocks[b]
        for hh in range(H):
            kb, vb = k_raw[toks, hh], v[toks, hh]
            wts = torch.softmax(kb @ pool_score[hh], 0)
            pk = wts @ kb
            target = (wts * kb.pow(2).mean(-1).add(1e-6).sqrt()).sum()
            kc[hh, j] = pk * target / pk.pow(2).mean().add(1e-6).sqrt()
            vc[hh, j] = wts @ vb
        centre = torch.tensor([b[0] * m + (m - 1) / 2]), torch.tensor([b[1] * m + (m - 1) / 2])
        cos, sin = rotary_angles(*centre, D, spec["rope_theta"])
        kc[:, j] = apply_rotary(kc[:, j], cos, sin)
    out = torch.zeros_like(q)
    for b in bkeys:
        group = blocks[b]
        S = torch.zeros(len(bkeys), dtype=q.dtype)
        for i in group:
            for hh in range(H):
                S += torch.softmax(q[i, hh] @ kc[hh].T * scale, 0)
        cands = [j for j, bb in enumerate(bkeys) if window_of(bb) != window_of(b)]
        chosen = sorted(cands, key=lambda j: -float(S[j]))[:n]
        sel = [t for j in chosen for t in blocks[bkeys[j]]]
        local = [t for t, (r, c) in enumerate(pos) if (r // W, c // W) == window_of(b)]
        for i in group:
            for hh in range(H):
                keys = torch.cat([kc[hh], k[sel + local, hh]])
                vals = torch.cat([vc[hh], v[sel + local, hh]])
                out[i, hh] = torch.softmax(keys @ q[i, hh] * scale, 0) @ vals
    return out


@pytest.mark.parametrize("h,w", [(10, 14), (7, 9), (4, 4), (2, 18)])
def test_sparse_2d_matches_its_definition(h, w):
    q, k_raw, v, pool = qkv(h * w)
    q, k = rotate(q, h, w), rotate(k_raw, h, w)
    got = sparse_attention_2d(q, k, v, k_raw, h, w, SPEC, pool)
    assert torch.allclose(got, reference_sparse_2d(q, k, v, k_raw, h, w, SPEC, pool), atol=1e-10)


def test_sparse_2d_batches_of_groups_and_their_checkpoints_change_nothing(monkeypatch):
    h, w = 10, 14
    q, k_raw, v, pool = qkv(h * w, seed=3)
    leaves = [x.clone().requires_grad_() for x in (q, k_raw, v, pool)]

    def run():
        qq, kr, vv, pp = leaves
        out = sparse_attention_2d(rotate(qq, h, w), rotate(kr, h, w), vv, kr, h, w, SPEC, pp)
        grads = torch.autograd.grad(out.pow(2).sum(), leaves)
        return out.detach(), grads

    whole = run()
    monkeypatch.setattr(vision, "BUDGET", 500)                      # a few query groups per batch, checkpointed
    split = run()
    assert torch.allclose(whole[0], split[0], atol=1e-12)
    assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(whole[1], split[1]))


def test_sparse_2d_gradcheck():
    h, w = 6, 6
    q, k_raw, v, pool = qkv(h * w, H=1, D=4, seed=5)
    spec = {**SPEC, "select": 1}
    f = lambda q, kr, v, p: sparse_attention_2d(rotate(q, h, w), rotate(kr, h, w), v, kr, h, w, spec, p)
    assert torch.autograd.gradcheck(f, tuple(x.requires_grad_() for x in (q, k_raw, v, pool)))


def test_window_attention_is_full_attention_inside_each_square():
    shapes = [(6, 9), (4, 4), (2, 2)]
    layout = Layout(torch.tensor(shapes), 4, "cpu")
    q, k, v, _ = qkv(layout.n)
    got = window_attention(q, k, v, layout)
    for (h, w), off in zip(shapes, layout.offsets):
        for i in range(h * w):
            r, c = divmod(i, w)
            keys = [off + t for t in range(h * w) if (t // w // 4, t % w // 4) == (r // 4, c // 4)]
            for hh in range(2):
                p = torch.softmax(k[keys, hh] @ q[off + i, hh] * 8**-0.5, 0)
                assert torch.allclose(got[off + i, hh], p @ v[keys, hh], atol=1e-12)


def test_global_layers_are_dense_up_to_dense_max_patches():
    layout = Layout(torch.tensor([[6, 9], [4, 4]]), 4, "cpu")
    q, k_raw, v, pool = qkv(layout.n)
    got = global_attention(q, k_raw, v, k_raw, layout, {**SPEC, "dense_max_patches": 54}, pool)
    sparse = global_attention(q, k_raw, v, k_raw, layout, {**SPEC, "dense_max_patches": 16}, pool)
    for (h, w), off in zip(layout.shapes, layout.offsets):
        sl = slice(off, off + h * w)
        full = torch.softmax(torch.einsum("thd,shd->hts", q[sl], k_raw[sl]) * 8**-0.5, -1)
        assert torch.allclose(got[sl], torch.einsum("hts,shd->thd", full, v[sl]), atol=1e-12)
    assert torch.equal(sparse[54:], got[54:]) and not torch.allclose(sparse[:54], got[:54])   # 6 x 9 > 16: sparse


def test_default_tower_has_four_global_layers_and_pool_scores_start_at_zero():
    model = tiny_model()
    model.add_vision(DEFAULT_VISION_CONFIG)
    layers = model.model.vision.tower.layers
    assert [i for i, l in enumerate(layers) if l.attn.is_global] == [3, 7, 11, 15][:len([l for l in layers if l.attn.is_global])]
    assert sum(l.attn.is_global for l in layers) == len(layers) // DEFAULT_VISION_CONFIG["tower"]["global_every"]
    assert all(l.attn.pool_score is None or not l.attn.pool_score.any() for l in layers)


def test_text_only_config_is_unchanged_and_vision_config_round_trips(tmp_path):
    assert "vision_config" not in BudgieConfig().to_dict()
    cfg = tiny_config(vision_config=TINY_VISION)
    cfg.save_pretrained(tmp_path)
    assert BudgieConfig.from_pretrained(tmp_path).vision_config == cfg.vision_config


@pytest.mark.parametrize("bad", [{"pixel_shuffle": 0}, {"min_image_tokens": 100}, {"image_token_id": 1},
                                 {"image_row_id": 18}, {"extra": 1}])
def test_bad_vision_config_is_refused(bad):
    with pytest.raises(ValueError):
        tiny_config(vision_config={**TINY_VISION, **bad})


def test_text_model_is_unchanged_by_add_vision():
    ids = image_ids(5, [], 6)
    a = tiny_model()(ids).logits
    m = tiny_model()
    m.add_vision(TINY_VISION)
    assert torch.equal(a, m(ids).logits)                         # same weights, no image: same output
    assert all(k.startswith("model.vision.") for k in set(m.state_dict()) - set(tiny_model().state_dict()))


def test_images_go_in_at_the_placeholders_and_only_affect_later_tokens():
    model = vision_model()
    grids = [(2, 3), (3, 1)]                                       # different shapes in one sequence
    ids = image_ids(4, grids, 5)
    with torch.no_grad():
        a = model(ids, **pixels(grids)).logits
        b = model(ids, **pixels(grids, seed=9)).logits
        c = model(ids).logits                                      # placeholders as plain token embeddings
    first = int((ids[0] == IMG).nonzero()[0])
    assert torch.equal(a[:, :first], b[:, :first])                # causal: nothing before the first image changes
    assert not torch.allclose(a[:, first:], b[:, first:]) and not torch.allclose(a, c)
    second = int((ids[0] == IMG).nonzero()[6])                    # the second image only reaches what comes after it
    p1, p2 = pixels(grids), pixels(grids)
    p2["pixel_values"][1] = pixels(grids, seed=11)["pixel_values"][1]
    with torch.no_grad():
        a, b = model(ids, **p1).logits, model(ids, **p2).logits
    assert torch.equal(a[:, :second], b[:, :second]) and not torch.allclose(a[:, second:], b[:, second:])


def test_projected_features_land_row_major_and_rows_tokens_stay_text():
    model = vision_model()
    grids = [(2, 3)]
    ids, px = image_ids(3, grids, 2), pixels(grids)
    x = model.model.embed(ids, None)
    out = model.model.put_images(x, ids, **px)
    feats = model.model.vision(**px)
    assert torch.allclose(out[0, ids[0] == IMG], feats) and torch.equal(out[0, ids[0] != IMG], x[0, ids[0] != IMG])
    # Token (r, c) is the 2 x 2 square of patch rows 2r..2r+1, cols 2c..2c+1, (y, x) order: fake a tower whose output
    # at patch k is k everywhere, and read what reaches the projector.
    vis, seen = model.model.vision, {}
    vis.tower.forward = lambda *a, **kw: torch.arange(24.0)[:, None].expand(24, 16)
    vis.proj_norm.register_forward_pre_hook(lambda m, args: seen.update(h=args[0]))
    vis(**px)
    got = seen["h"][:, ::16].tolist()                              # the first channel of each of the 4 patches
    want = [[(2 * r) * 6 + 2 * c, (2 * r) * 6 + 2 * c + 1, (2 * r + 1) * 6 + 2 * c, (2 * r + 1) * 6 + 2 * c + 1]
            for r in range(2) for c in range(3)]
    assert got == want


def test_padding_does_not_leak_into_an_image_inside_the_tower():
    model = vision_model()
    tower = model.model.vision.tower
    grids = [(2, 3), (1, 2)]                                      # 24 and 8 patches; image 1 is padded to 24
    px = pixels(grids)
    with torch.no_grad():
        both = tower(**px)                                        # packed: image 0's 24 patches, then image 1's 8
        px2 = {k: v.clone() for k, v in px.items()}
        px2["pixel_values"][1, 8:] = 123.0                         # garbage in image 1's padding
        assert both.shape[0] == 32 and torch.equal(both, tower(**px2))
        alone = tower(px["pixel_values"][1:, :8], px["pixel_attention_mask"][1:, :8], px["spatial_shapes"][1:])
    assert torch.allclose(both[24:], alone, atol=1e-6)


def test_tower_gradient_checkpointing_gives_the_same_gradients():
    grads = []
    for ckpt in (False, True):
        model = vision_model().train()
        model.model.vision.tower.gradient_checkpointing = ckpt
        grids = [(2, 2)]
        ids = image_ids(3, grids, 3)
        model(ids, **pixels(grids), labels=ids).loss.backward()
        grads.append(model.model.vision.tower.stem.proj[0].weight.grad.clone())
    assert torch.allclose(*grads, atol=1e-6)


def test_padding_and_batching_do_not_change_an_image():
    model = vision_model()
    grids = [(2, 3), (1, 2)]
    ids = torch.cat([image_ids(3, grids[:1], 4, seed=5), image_ids(3, grids[1:], 9, seed=6)])
    px = pixels(grids)
    alone = {"pixel_values": px["pixel_values"][1:, :8], "pixel_attention_mask": px["pixel_attention_mask"][1:, :8],
             "spatial_shapes": px["spatial_shapes"][1:]}
    with torch.no_grad():
        both = model(ids, **px).logits
        row1 = model(ids[1:], **alone).logits
    assert torch.allclose(both[1:], row1, atol=1e-5)


def test_wrong_image_count_and_missing_inputs_are_refused():
    model = vision_model()
    with pytest.raises(ValueError, match="images need"):
        model(image_ids(3, [(2, 2), (2, 2)], 2), **pixels([(2, 2)]))
    with pytest.raises(ValueError, match="images need"):
        model(image_ids(3, [(2, 2)], 2), **pixels([(1, 2)]))
    with pytest.raises(ValueError, match="spatial_shapes"):
        model(image_ids(3, [(2, 2)], 2), pixel_values=pixels([(2, 2)])["pixel_values"])
    with pytest.raises(ValueError, match="no vision_config"):
        tiny_model()(image_ids(3, [(2, 2)], 2), **pixels([(2, 2)]))


def test_image_span_is_never_predicted():
    model = vision_model()
    grids = [(2, 2)]
    ids = image_ids(4, grids, 5)
    masked = ids.masked_fill(torch.isin(ids, torch.tensor([IMG, START, END, ROW])), -100)
    a = model(ids, **pixels(grids), labels=ids).loss
    b = model(ids, **pixels(grids), labels=masked).loss
    assert torch.equal(a, b)


def test_gradients_reach_the_projector_and_tower():
    model = vision_model().train()
    grids = [(2, 3)]
    ids = image_ids(3, grids, 4)
    model(ids, **pixels(grids), labels=ids).loss.backward()
    v = model.model.vision
    assert v.fc1.weight.grad.abs().sum() > 0 and v.tower.stem.proj[0].weight.grad.abs().sum() > 0


def test_generate_with_an_image_matches_the_uncached_forward():
    model = vision_model()
    grids = [(2, 3)]
    ids, px = image_ids(4, grids, 3), pixels(grids)
    out = model.generate(ids, **px, max_new_tokens=5, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 5
    with torch.no_grad():                                          # greedy steps, recomputed without a cache
        seq = ids
        for _ in range(5):
            nxt = model(seq, **px, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], 1)
    assert torch.equal(out, seq)


def test_save_and_load_with_vision(tmp_path):
    model = vision_model()
    model.save_pretrained(tmp_path)
    back = BudgieForCausalLM.from_pretrained(tmp_path).eval()
    grids = [(3, 2)]
    ids = image_ids(3, grids, 3)
    with torch.no_grad():
        assert torch.equal(model(ids, **pixels(grids)).logits, back(ids, **pixels(grids)).logits)


def test_fit_image_keeps_aspect_and_budget():
    assert fit_image(1080, 1920) == (34, 60)                      # a 1080p screenshot at full resolution: 2040 tokens
    assert fit_image(2400, 1080) == (67, 30)                      # a phone screenshot, portrait
    assert fit_image(40, 40) == (8, 8)                            # an icon is scaled up to the 64-token minimum
    rng = random.Random(0)
    for _ in range(500):
        h, w = rng.randint(1, 5000), rng.randint(1, 5000)
        if max(h, w) / min(h, w) > 50:
            continue
        r, c = fit_image(h, w)
        assert r * c <= 2048 and (r * c >= 64 or r == 1 or c == 1)
        assert abs(np.log((r / c) / (h / w))) < np.log(1 + 2 / min(r, c)) + 0.05   # aspect ratio kept, to the rounding
    with pytest.raises(ValueError):
        fit_image(10, 5000)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(ROOT / "tokenizer")


def test_image_tokens_are_single_ids(tok):
    assert tok.convert_tokens_to_ids([IMAGE, IMAGE_START, IMAGE_END, IMAGE_ROW]) == [IMG, START, END, ROW]
    assert tok.encode(image_span(2, 3), add_special_tokens=False) == span_ids(2, 3)
    assert IMG not in tok.encode("<image> an <img> tag", add_special_tokens=False)    # plain text never makes one


def test_chat_template_writes_one_placeholder_per_image(tok):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "and"}, {"type": "image"}]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert text == "<bos><|user|>\n<|image|>and<|image|><|end|>\n<|assistant|>\n<think>\n"


def test_processor_sizes_expands_and_feeds_the_model(tok):
    proc = BudgieProcessor(tok, tiny_config(vision_config=TINY_VISION))
    rng = np.random.default_rng(0)
    wide = Image.fromarray(rng.integers(0, 255, (50, 110, 3), dtype=np.uint8))       # 16 px tokens: 3 x 7
    tall = Image.fromarray(rng.integers(0, 255, (64, 30, 3), dtype=np.uint8)).convert("L")
    out = proc(f"a {IMAGE} b {IMAGE}", images=[wide, tall])
    grids = [proc.grid(wide), proc.grid(tall)]
    assert grids == [(3, 7), (4, 2)]
    ids = out["input_ids"][0].tolist()
    assert ids.count(IMG) == 3 * 7 + 4 * 2 and ids.count(ROW) == 3 + 4 and ids.count(START) == ids.count(END) == 2 and ids[0] == 1
    assert out["spatial_shapes"].tolist() == [[6, 14], [8, 4]] and out["pixel_attention_mask"].sum(1).tolist() == [84, 32]
    assert out["pixel_values"].shape == (2, 84, PATCH_DIM) and -1 <= out["pixel_values"].min() and out["pixel_values"].max() <= 1
    assert proc.grid(wide, max_tokens=8) == (1, 4)                # a per-call budget
    assert collapse_images(proc.tokenizer.decode(ids)) == f"<bos>a {IMAGE} b {IMAGE}"
    with pytest.raises(ValueError):
        proc(f"a {IMAGE}", images=[])
    with pytest.raises(ValueError):
        expand_images(f"{IMAGE_START}", [])
    model = vision_model()
    with torch.no_grad():                                         # straight into the model (tiny vocab; image ids kept)
        model(out["input_ids"].clamp(max=VOCAB - 1), **{k: out[k] for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")})
