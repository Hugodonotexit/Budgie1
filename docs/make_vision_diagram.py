"""Draws docs/vision.svg from DEFAULT_VISION_CONFIG (configuration_budgie.py) and config.json. Standard library only:
    python docs/make_vision_diagram.py

Three parts: how an image becomes tokens of the text model (left), what the tower's two attention kinds read
(right, top), and how the Budgie 1 tower is trained (right, bottom). Widths, layer counts, windows and budgets come from
the code, so re-run this after changing them."""

import ast
import json
import pathlib
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "configuration_budgie.py").read_text()
tree = ast.parse(src)
V = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
         and any(getattr(t, "id", None) == "DEFAULT_VISION_CONFIG" for t in n.targets))
T = V["tower"]
cfg = json.loads((ROOT / "config.json").read_text())

W, H = 1040, 1000
INK, MUTED, BG = "#0f172a", "#475569", "#ffffff"
STEM, TOWER, WIN, GLOB, PROJ, SEQ, TRAIN = (("#e0f2fe", "#0284c7"), ("#f8fafc", "#94a3b8"), ("#dbeafe", "#2563eb"),
                                           ("#cffafe", "#0891b2"), ("#ede9fe", "#7c3aed"), ("#fef3c7", "#d97706"),
                                           ("#dcfce7", "#16a34a"))
PLAIN = ("#f1f5f9", "#64748b")
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
out, problems = [], []


def rect(x, y, w, h, colors=PLAIN, rx=8, dash=False, sw=1.5):
    fill, stroke = colors
    d = ' stroke-dasharray="6 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=13, weight="normal", anchor="middle", fill=INK, fit=None):
    if fit and len(s) * size * 0.55 > fit:
        problems.append(f"text too wide for {fit}px: {s!r}")
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(s)}</text>')


def box(x, y, w, h, lines, colors=PLAIN, size=12.5, weight="bold", rx=8, dash=False):
    rect(x, y, w, h, colors, rx, dash)
    n = len(lines)
    first = y + h / 2 - (n - 1) * (size + 3) / 2 + size * 0.35
    for i, line in enumerate(lines):
        text(x + w / 2, first + i * (size + 3), line, size, weight if i == 0 else "normal", fit=w - 10)


def arrow(x1, y1, x2, y2, color=MUTED):
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" marker-end="url(#a)"/>')


def k(n):
    return f"{n // 1024}k" if n % 1024 == 0 and n >= 1024 else f"{n:,}"


d, p, s = T["hidden_size"], T["patch_size"], V["pixel_shuffle"]
unit = p * s
widths = [T["num_channels"], *(T["stem"] or []), d]
n_layers, every = T["num_hidden_layers"], T["global_every"]
glob = [i + 1 for i in range(n_layers) if (i + 1) % every == 0]
ffn = " → ".join(str(x) for x in [d, *T["intermediate_size"], d])
lm = cfg["hidden_size"]

text(32, 34, f"Budgie image input: {n_layers}-layer vision tower, width {d}, one token per {unit} × {unit} px", 18, "bold", "start")

# ---------------------------------------------------------------- left: image -> tokens
LX, LW, CX = 32, 470, 32 + 235
y = 62
box(LX, y, LW, 56, ["Image, any aspect ratio",
                    f"resized to whole {unit}-px tokens, {V['min_image_tokens']}–{V['max_image_tokens']:,} tokens in all",
                    "(1920 × 1080 stays native: 34 × 60 = 2,040 tokens)"], PLAIN, 12)
arrow(CX, y + 56, CX, y + 78)
y += 78
if T["stem"]:
    box(LX, y, LW, 58, [f"Conv stem: {len(widths) - 1} stride-2 stages",
                        f"{T['conv']}×{T['conv']} depthwise conv → 2×2 space-to-depth → Linear",
                        "widths " + " → ".join(str(x) for x in widths) + f": one vector per {p} × {p} px patch"], STEM, 12)
else:
    box(LX, y, LW, 44, ["Linear patch embedding", f"{p} × {p} × 3 → {d}"], STEM, 12)
arrow(CX, y + 58, CX, y + 80)
y += 80
ty = y
th = 326
rect(LX, ty, LW, th, TOWER, 12, dash=True)
text(LX + 14, ty + 24, f"BudgieViT × {n_layers} layers", 14, "bold", "start")
text(LX + LW - 14, ty + 24, "patches of all images packed: no padding", 11.5, "normal", "end", MUTED)
iy = ty + 40
box(LX + 16, iy, LW - 32, 30, ["RMSNorm → attention → + x"], PLAIN, 12, "normal")
box(LX + 16, iy + 36, (LW - 40) / 2, 66, ["Windowed",
                                          f"{T['window']} × {T['window']} patch squares",
                                          f"layers 1–{n_layers} but {', '.join(map(str, glob))}"], WIN, 11.5)
box(LX + 24 + (LW - 40) / 2, iy + 36, (LW - 40) / 2, 66, ["Global (every " + str(every) + "th)",
                                                       f"dense up to {T['dense_max_patches']:,} patches,",
                                                       "else compressed + selected"], GLOB, 11.5)
text(CX, iy + 120, f"QK-norm · 2-D RoPE (half the dims by row, half by column) · {T['num_attention_heads']} heads × {T['head_dim']}",
     11.5, fill=MUTED, fit=LW - 20)
text(CX, iy + 136, f"Q and K/V each read a {T['conv']}×{T['conv']} depthwise conv over the image's patch grid", 11.5, fill=MUTED, fit=LW - 20)
box(LX + 16, iy + 150, LW - 32, 30, ["RMSNorm → feed-forward → + x"], PLAIN, 12, "normal")
text(CX, iy + 198, f"{T['conv']}×{T['conv']} depthwise conv, then {ffn}", 11.5, fill=MUTED, fit=LW - 20)
text(CX, iy + 214, "SiLU / centred dSiLU / SiLU, as in the text model", 11.5, fill=MUTED, fit=LW - 20)
box(LX + 16, iy + 226, LW - 32, 26, ["final RMSNorm"], PLAIN, 12, "normal")
text(CX, ty + th - 12, "masked training (keep=): only kept patches run, convs see zeros",
     11, fill=MUTED)
y = ty + th
arrow(CX, y, CX, y + 22)
y += 22
box(LX, y, LW, 44, [f"Pixel shuffle {s} × {s}", f"each {s}×{s} square of patch outputs → one {s * s * d}-wide vector"], PROJ, 12)
arrow(CX, y + 44, CX, y + 66)
y += 66
box(LX, y, LW, 44, ["Projector", f"LayerNorm → Linear {s * s * d} → {lm} → GELU → Linear {lm} → {lm}"], PROJ, 12)
arrow(CX, y + 44, CX, y + 66)
y += 66
box(LX, y, LW, 74, ["In the text model's sequence",
                    "<|image_start|>  ( cols × <|image|>  <|image_row|> ) × rows  <|image_end|>",
                    "the <|image|> embeddings are replaced by the projected vectors,",
                    "row-major; nothing in the span is ever predicted"], SEQ, 11.5)
left_bottom = y + 74

# ---------------------------------------------------------------- right top: attention schematic
RX, RW = 540, 468
rect(RX, 62, RW, 330, ("#fafafa", "#cbd5e1"), 12)
text(RX + 14, 86, "What one patch's query reads", 14, "bold", "start")
text(RX + RW - 14, 86, "schematic: a 16 × 24 patch grid", 11.5, "normal", "end", MUTED)
CELL, GR, GC = 9, 16, 24
win = 8                                          # drawn window side (schematic: the real one is T["window"] patches)


def grid(x0, y0, colour_of):
    for r in range(GR):
        for c in range(GC):
            col = colour_of(r, c)
            out.append(f'<rect x="{x0 + c * CELL}" y="{y0 + r * CELL}" width="{CELL - 1}" height="{CELL - 1}" fill="{col}"/>')


q = (5, 9)
gx1, gy = RX + 14, 104
grid(gx1, gy, lambda r, c: "#dc2626" if (r, c) == q else WIN[1] if r // win == q[0] // win and c // win == q[1] // win else "#e2e8f0")
text(gx1 + GC * CELL / 2, gy + GR * CELL + 18, "windowed: its own square only", 11.5, fill=MUTED)
gx2 = RX + RW - 14 - GC * CELL
sel = {(0, 4), (3, 1), (2, 5)}                   # blocks of 4 x 4 patches chosen by the compressed scores


def sparse_colour(r, c):
    if (r, c) == q:
        return "#dc2626"
    if r // win == q[0] // win and c // win == q[1] // win:
        return WIN[1]
    if (r // 4, c // 4) in sel:
        return GLOB[1]
    return "#a5f3fc" if (r % 4 == 1 and c % 4 == 1) else "#e2e8f0"


grid(gx2, gy, sparse_colour)
text(gx2 + GC * CELL / 2, gy + GR * CELL + 18, "global, large image", 11.5, fill=MUTED)
text(gx2 + GC * CELL / 2, gy + GR * CELL + 33, "own square + chosen + pooled blocks", 11.5, fill=MUTED)
ly = gy + GR * CELL + 56
for i, (col, label) in enumerate([("#dc2626", "the query's patch"), (WIN[1], f"its {T['window']} × {T['window']}-patch square, exactly"),
                                  (GLOB[1], f"top-{T['select']} blocks by compressed score, exactly"),
                                  ("#a5f3fc", f"every {T['compress']}×{T['compress']} block as one pooled key")]):
    out.append(f'<rect x="{RX + 20}" y="{ly + i * 20 - 10}" width="12" height="12" fill="{col}"/>')
    text(RX + 40, ly + i * 20, label, 11.5, anchor="start", fill=INK)
text(RX + 20, ly + 84, f"Up to {T['dense_max_patches']:,} patches the global layers use plain full attention.", 11.5,
     anchor="start", fill=MUTED)

# ---------------------------------------------------------------- right bottom: training
ry = 412
rect(RX, ry, RW, left_bottom - ry, ("#fafafa", "#cbd5e1"), 12)
text(RX + 14, ry + 24, "How the Budgie 1 tower is trained", 14, "bold", "start")
steps = [
    ("1–3  Alone, on the CPU: masked autoencoder", ["sees 25% of the patches, a small decoder rebuilds the rest",
                                                 "128 px → 256 px → 512 px, square crops (70 / 20 / 10%)"]),
    ("4a  Joins the text model: projector only", ["text model and tower frozen; captions and page text,",
                                                 "images at most 1,024 tokens"]),
    ("4b  Continued pretraining, everything trains", ["images 20% of tokens at 2k → 5% at 256k; budget",
                                                     "1,024 tokens at 2k, 2,048 at 4k, 4,096 from 8k"]),
    ("4c  Anneal and long context", ["6% high-quality image data; long agent runs at 256k"]),
]
sy = ry + 40
for title, lines in steps:
    h = 22 + 16 * len(lines)
    rect(RX + 14, sy, RW - 28, h, TRAIN, 8)
    text(RX + 26, sy + 17, title, 12.5, "bold", "start", fit=RW - 50)
    for j, line in enumerate(lines):
        text(RX + 26, sy + 34 + j * 16, line, 11.5, "normal", "start", MUTED, fit=RW - 50)
    sy += h + 10
H = max(left_bottom, sy) + 24

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
       '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0 0L10 5L0 10z" fill="{MUTED}"/></marker></defs>', f'<rect width="{W}" height="{H}" fill="{BG}"/>', *out, "</svg>"]
(ROOT / "docs" / "vision.svg").write_text("\n".join(svg) + "\n")
print(f"wrote docs/vision.svg ({W} x {H})")
for msg in problems:
    print("WARNING", msg)
