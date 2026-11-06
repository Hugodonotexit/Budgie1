"""BudgieProcessor: text + images -> input_ids, attention_mask, pixel_values, pixel_attention_mask, spatial_shapes.

Text marks each image with one <|image|> (the chat template writes one per {"type": "image"} part). Each image is
resized, keeping its aspect ratio, to whole 32 x 32 px tokens (patch 16 x pixel_shuffle 2) with between
min_image_tokens and max_image_tokens in all (64..4096 by default; Qwen3-VL's smart_resize rule), so a 1920 x 1080
screenshot stays at full resolution (60 x 34 tokens). Its <|image|> becomes

    <|image_start|> (cols x <|image|> <|image_row|>) x rows <|image_end|>

and its pixels become rows*2 x cols*2 patches of 16 x 16 x 3 in [-1, 1], padded to the
batch's largest image.

    proc = BudgieProcessor.from_pretrained("path/to/checkpoint")
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Click Save."}]}],
                                    add_generation_prompt=True)
    inputs = proc(text, images=[screenshot], add_special_tokens=False)   # the template already starts with <bos>
    model.generate(**inputs, max_new_tokens=200)
"""

import math
import re

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from .configuration_budgie import BudgieConfig

IMAGE, IMAGE_START, IMAGE_END, IMAGE_ROW = "<|image|>", "<|image_start|>", "<|image_end|>", "<|image_row|>"
MAX_ASPECT = 200


def fit_image(height, width, unit=32, min_tokens=64, max_tokens=2048):
    """(rows, cols) of unit x unit px tokens for an image of height x width px: each side rounded to whole units, then
    scaled (aspect ratio kept) into [min_tokens, max_tokens] tokens in all. Qwen2-VL / Qwen3-VL's smart_resize."""
    if height < 1 or width < 1 or max(height, width) / min(height, width) > MAX_ASPECT:
        raise ValueError(f"image {width} x {height}: sides must be >= 1 px with aspect ratio at most {MAX_ASPECT}:1")
    rows, cols = max(1, round(height / unit)), max(1, round(width / unit))
    if rows * cols > max_tokens:
        beta = math.sqrt(height * width / (max_tokens * unit * unit))
        rows, cols = max(1, math.floor(height / beta / unit)), max(1, math.floor(width / beta / unit))
    elif rows * cols < min_tokens:
        beta = math.sqrt(min_tokens * unit * unit / (height * width))
        rows, cols = math.ceil(height * beta / unit), math.ceil(width * beta / unit)
    return rows, cols


def image_span(rows, cols):
    """The placeholder text of one image of rows x cols tokens."""
    return IMAGE_START + (IMAGE * cols + IMAGE_ROW) * rows + IMAGE_END


def expand_images(text, grids):
    """Each <|image|> of text -> image_span of the next (rows, cols) of grids, in order."""
    if IMAGE_START in text:
        raise ValueError(f"text already has {IMAGE_START}: give each image as a single {IMAGE}")
    parts = text.split(IMAGE)
    if len(parts) - 1 != len(grids):
        raise ValueError(f"the text has {len(parts) - 1} {IMAGE} but {len(grids)} images were given")
    return parts[0] + "".join(image_span(*g) + p for g, p in zip(grids, parts[1:]))


def collapse_images(text):
    """Text with every image span collapsed back to one <|image|> (for printing a prompt)."""
    body = f"(?:{re.escape(IMAGE)}|{re.escape(IMAGE_ROW)})*"
    return re.sub(re.escape(IMAGE_START) + body + re.escape(IMAGE_END), IMAGE, text)


def image_patches(image, rows, cols, patch=16, shuffle=2):
    """A PIL image -> [rows*shuffle * cols*shuffle, patch*patch*3] float patches in [-1, 1], row-major, each patch
    (y, x, channel) flattened."""
    h, w = rows * shuffle * patch, cols * shuffle * patch
    img = image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0   # (x / 255 - 0.5) / 0.5
    gh, gw = h // patch, w // patch
    return x.reshape(3, gh, patch, gw, patch).permute(1, 3, 2, 4, 0).reshape(gh * gw, patch * patch * 3)


class BudgieProcessor:
    def __init__(self, tokenizer, config: BudgieConfig):
        if not config.has_vision:
            raise ValueError("this config has no vision_config")
        self.tokenizer, self.config = tokenizer, config
        v = config.vision_config
        self.patch, self.shuffle = v["tower"]["patch_size"], v["pixel_shuffle"]
        self.min_tokens, self.max_tokens = v["min_image_tokens"], v["max_image_tokens"]
        got = tokenizer.convert_tokens_to_ids([IMAGE, IMAGE_START, IMAGE_END, IMAGE_ROW])
        if got != [v["image_token_id"], v["image_start_id"], v["image_end_id"], v["image_row_id"]]:
            raise ValueError(f"tokenizer ids {got} for {IMAGE}/{IMAGE_START}/{IMAGE_END}/{IMAGE_ROW} differ from vision_config's")

    @classmethod
    def from_pretrained(cls, path, tokenizer_path=None):
        """A checkpoint directory: config.json there, the tokenizer there or in its tokenizer/ subdirectory."""
        config = BudgieConfig.from_pretrained(path)
        try:
            tok = AutoTokenizer.from_pretrained(tokenizer_path or path)
        except (OSError, ValueError):
            tok = AutoTokenizer.from_pretrained(f"{path}/tokenizer")
        return cls(tok, config)

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("tokenize", False)
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def grid(self, image, max_tokens=None, min_tokens=None):
        """(rows, cols) of tokens a PIL image becomes."""
        return fit_image(image.height, image.width, self.patch * self.shuffle,
                         min_tokens or self.min_tokens, max_tokens or self.max_tokens)

    def images(self, images, max_tokens=None, min_tokens=None):
        """PIL images -> (grids, {pixel_values [n, max_patches, patch*patch*3], pixel_attention_mask [n, max_patches],
        spatial_shapes [n, 2] patch rows / cols})."""
        grids = [self.grid(im, max_tokens, min_tokens) for im in images]
        patches = [image_patches(im, r, c, self.patch, self.shuffle) for im, (r, c) in zip(images, grids)]
        longest = max(len(p) for p in patches)
        pixels = torch.zeros(len(patches), longest, patches[0].shape[1])
        mask = torch.zeros(len(patches), longest, dtype=torch.int32)
        for i, p in enumerate(patches):
            pixels[i, :len(p)], mask[i, :len(p)] = p, 1
        shapes = torch.tensor([[r * self.shuffle, c * self.shuffle] for r, c in grids], dtype=torch.long)
        return grids, {"pixel_values": pixels, "pixel_attention_mask": mask, "spatial_shapes": shapes}

    def __call__(self, text, images=None, max_image_tokens=None, min_image_tokens=None, return_tensors="pt", **kwargs):
        """text: a string or a list of strings (a batch; pass padding=True for unequal lengths). images: every image
        of the batch in order of appearance (rows in order), one per <|image|>. max_image_tokens / min_image_tokens
        override the config's budget for this call (e.g. a smaller budget for photos, a larger one for documents)."""
        texts = [text] if isinstance(text, str) else list(text)
        images = [] if images is None else (list(images) if isinstance(images, (list, tuple)) else [images])
        n = sum(t.count(IMAGE) for t in texts)
        if n != len(images):
            raise ValueError(f"the text has {n} {IMAGE} but {len(images)} images were given")
        grids, pixels = self.images(images, max_image_tokens, min_image_tokens) if images else ([], {})
        expanded, k = [], 0
        for t in texts:
            m = t.count(IMAGE)
            expanded.append(expand_images(t, grids[k:k + m]))
            k += m
        out = self.tokenizer(expanded[0] if isinstance(text, str) else expanded, return_tensors=return_tensors, **kwargs)
        out.update(pixels)
        return out

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
