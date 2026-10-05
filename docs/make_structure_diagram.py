"""Draws docs/structure.svg from config.json. Standard library only:  python docs/make_structure_diagram.py

The picture has three parts: the whole stack (left), one decoder layer (right), and which earlier tokens
the newest token attends to in each layer kind (bottom). Layer counts, windows, dilations and widths come
from config.json, so re-run this after changing the defaults."""

import json
import pathlib
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / "config.json").read_text())

W, H = 1040, 1196
INK, MUTED, BG, GRID = "#0f172a", "#475569", "#ffffff", "#e2e8f0"
KIND = {  # fill, stroke
    "S": ("#dbeafe", "#2563eb"),
    "D4": ("#dcfce7", "#16a34a"),
    "D8": ("#fef3c7", "#d97706"),
    "G": ("#fce7f3", "#db2777"),
}
BRANCH = ("#ede9fe", "#7c3aed")
PLAIN = ("#f1f5f9", "#64748b")
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
out, problems = [], []


def rect(x, y, w, h, colors=PLAIN, rx=8, dash=False, sw=1.5):
    fill, stroke = colors
    d = ' stroke-dasharray="6 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=13, weight="normal", anchor="middle", fill=INK, style="normal", fit=None):
    if fit and len(s) * size * 0.6 > fit:
        problems.append(f"text too wide for {fit}px: {s!r}")
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" fill="{fill}">{escape(s)}</text>')


def box(x, y, w, h, lines, colors=PLAIN, size=13, weight="normal", rx=8, dash=False):
    """A box with one or more centred lines of text."""
    rect(x, y, w, h, colors, rx, dash)
    n = len(lines)
    first = y + h / 2 - (n - 1) * (size + 3) / 2 + size * 0.35
    for i, line in enumerate(lines):
        text(x + w / 2, first + i * (size + 3), line, size, weight if i == 0 else "normal", fit=w - 8)


def arrow(x1, y1, x2, y2, color=MUTED, dash=False, head=True):
    d = ' stroke-dasharray="5 4"' if dash else ""
    m = ' marker-end="url(#a)"' if head else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{d}{m}/>')


def path(d, color=MUTED, dash=False, head=True):
    ds = ' stroke-dasharray="5 4"' if dash else ""
    m = ' marker-end="url(#a)"' if head else ""
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"{ds}{m}/>')


def plus(x, y):
    out.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{BG}" stroke="{MUTED}" stroke-width="1.6"/>')
    out.append(f'<path d="M{x-5} {y}H{x+5}M{x} {y-5}V{y+5}" stroke="{MUTED}" stroke-width="1.8"/>')


def k(n):
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


# ---- numbers from config.json ------------------------------------------------------------------
d = cfg["hidden_size"]
pattern, blocks = cfg["block_pattern"], cfg["num_blocks"]
types = cfg["attention_types"]
heads, hdim = cfg["num_attention_heads"], cfg["head_dim"]
ffn = cfg["intermediate_size"]
cuts, div = cfg["adaptive_cutoffs"], cfg["adaptive_div"]
branch_on = cfg.get("linear_branch", "none") != "none"
reach = lambda t: None if types[t]["window"] is None else types[t]["window"] * types[t]["dilation"]
shared = sorted({int(r) // len(pattern) + 1 for r in cfg["kv_share"]})

out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">')
out.append('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M0 0L10 5L0 10z" fill="{MUTED}"/></marker></defs>')
out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
text(30, 34, f"Budgie: {blocks} blocks × {len(pattern)} layers = {blocks * len(pattern)} layers, hidden size {d}", 18, "bold", "start")

# ---- left: the whole stack ---------------------------------------------------------------------
CX = 288
text(CX, 70, "Token ids", 13, fill=MUTED)
arrow(CX, 78, CX, 100)
text(CX, 118, "Input embedding", 14, "bold")
ew = [d // div**i for i in range(len(cuts) + 1)]
box(66, 130, 222, 76, ["Adaptive embedding", f"frequency rank clusters", f"cut at {cuts[0]} / {cuts[1]}" if len(cuts) > 1 else f"cut at {cuts[0]}",
                        f"widths {' / '.join(map(str, ew))}"], PLAIN, 12)
box(300, 130, 222, 76, ["Hashed n-gram tables", f"orders {', '.join(map(str, cfg['ngram_orders']))}, {cfg['ngram_heads']} hash heads", f"{cfg['ngram_buckets']:,} rows × {cfg['ngram_dim']}",
                        "→ Linear up to d"], PLAIN, 12)
plus(CX, 232)
path(f"M177 206V232H{CX-11}", head=False)
path(f"M411 206V232H{CX+11}", head=False)
arrow(CX, 243, CX, 268)

BX, BY, BW = 56, 268, 464
rows = len(pattern)
RH, GAP = 40, 10
by_end = BY + 36 + rows * (RH + GAP) + 6
rect(BX, BY, BW, by_end - BY, ("#fafafa", "#94a3b8"), 12, dash=True)
text(BX + 14, BY + 24, f"Block × {blocks}", 14, "bold", "start")
text(BX + BW - 14, BY + 24, "same pattern, separate weights", 12, "normal", "end", MUTED)
LX, LW = 174, 240
row_y = {}
for i, kind in enumerate(pattern):
    y = BY + 36 + i * (RH + GAP)
    row_y[i] = y
    t = types[kind]
    desc = "global, no positions" if t["window"] is None else f"window {k(t['window'])}" + (f", stride {t['dilation']}" if t["dilation"] > 1 else "")
    box(LX, y, LW, RH, [f"{kind}   {desc}"], KIND[kind], 13)
    if i:
        arrow(LX + LW / 2, row_y[i - 1] + RH, LX + LW / 2, y)
    tag = ("reach " + k(reach(kind))) if reach(kind) else "reach: all"
    text(LX + LW + 8, y + RH / 2 + 4, tag + (" · KV*" if kind in ("D4", "D8", "G") and shared else ""), 11, "normal", "start", MUTED)
g_first = pattern.index("G")
if branch_on:
    lh, lw = cfg["lin_heads"], cfg["lin_head_dim"]
    ly0, ly1 = row_y[0] + 4, row_y[g_first] - 12
    box(BX + 8, row_y[1], 100, row_y[3] + RH - row_y[1], ["Retention", "branch", f"{lh} heads × {lw}", "fixed decay", "reset per doc"], BRANCH, 11.5)
    path(f"M{BX+58} {BY+30}V{row_y[1]}", BRANCH[1])
    path(f"M{BX+58} {row_y[3]+RH}V{row_y[g_first]-8}H{LX-13}", BRANCH[1])
    plus(LX - 24, row_y[g_first] - 8)
    text(BX + 8, row_y[g_first] + 18, "gated output", 11, "normal", "start", BRANCH[1])
ft = by_end + 12
text(BX, ft + 6, "*KV: " + ", ".join(map(str, shared)) + f" blocks reuse the K/V of the block before them", 11.5, "normal", "start", MUTED)
arrow(CX, by_end, CX, by_end + 52 + 8, head=True)  # to final norm
box(CX - 80, by_end + 60, 160, 32, ["RMSNorm"], PLAIN, 13)
arrow(CX, by_end + 92, CX, by_end + 120)
hy = by_end + 120
box(66, hy, 456, 66, ["Adaptive softmax head, tied to the embedding tables",
                      f"logit soft-cap {cfg['logit_softcap']:g}, loss computed in chunks"], PLAIN, 12.5)
arrow(CX, hy + 66, CX, hy + 94)
text(CX, hy + 112, f"log-probabilities over {cfg['vocab_size']:,} tokens (or the loss, given labels)", 12.5, fill=MUTED)
# tied weights: a dashed line from the head back up to the embedding
path(f"M66 {hy+33}H34V168H66", MUTED, dash=True)
out.append(f'<text transform="translate(24 {hy-60}) rotate(-90)" font-size="11.5" fill="{MUTED}" text-anchor="middle">tied weights</text>')
left_end = hy + 120

# ---- right: one decoder layer ------------------------------------------------------------------
PX = 556
RX = PX + 18          # residual rail
RCX = PX + 219        # centre of the column
text(PX, 70, "One layer", 14, "bold", "start")
text(PX + 100, 70, "x = x + attention(norm(x));  x = x + ffn(norm(x))", 11.5, "normal", "start", MUTED)
rail_x = 1024
arrow(RCX, 82, RCX, 96)
box(RCX - 80, 96, 160, 28, ["RMSNorm"], PLAIN, 13)
arrow(RCX, 124, RCX, 140)
AY = 140
q, kv = (PX, 218), (PX + 238, 218)
rect(PX - 6, AY, 450, 296, ("#fafafa", "#94a3b8"), 12, dash=True)
text(PX + 6, AY + 22, "Attention", 14, "bold", "start")
text(PX + 438, AY + 22, f"{heads} query heads × {hdim}", 12, "normal", "end", MUTED)
col_w = 212
for cx0, title, items in (
    (PX + 2, "Q", ["causal conv (2–4 taps)", f"Wq", "QK-norm", "RoPE (not in G)"]),
    (PX + 224, "K, V   (own layers only)", ["causal conv (own)", "Wk, Wv (grouped heads)", "QK-norm", "RoPE (not in G)"]),
):
    text(cx0 + col_w / 2, AY + 46, title, 12.5, "bold", fill=MUTED)
    for j, label in enumerate(items):
        box(cx0, AY + 56 + j * 38, col_w, 30, [label], PLAIN, 12.5)
        if j:
            arrow(cx0 + col_w / 2, AY + 56 + (j - 1) * 38 + 30, cx0 + col_w / 2, AY + 56 + j * 38)
sy = AY + 56 + 4 * 38 + 6
box(PX + 2, sy, 434, 46, ["Scaled dot-product attention", "windowed / dilated / global, + a learned sink per head"], KIND["S"], 12.5)
for cx0 in (PX + 2 + col_w / 2, PX + 224 + col_w / 2):
    arrow(cx0, AY + 56 + 3 * 38 + 30, cx0, sy)
arrow(RCX, sy + 46, RCX, sy + 58)
box(RCX - 100, sy + 58, 200, 28, ["Wo"], PLAIN, 13)
att_end = AY + 296
arrow(RCX, att_end, RCX, att_end + 18)
plus(RCX, att_end + 29)
# residual rail around the attention
path(f"M{RCX} 82H{rail_x}V{att_end+29}H{RCX+11}", MUTED, dash=True)
text(rail_x - 6, 98, "residual", 11, "normal", "end", MUTED)
n2 = att_end + 52
arrow(RCX, att_end + 40, RCX, n2)
box(RCX - 80, n2, 160, 28, ["RMSNorm"], PLAIN, 13)
arrow(RCX, n2 + 28, RCX, n2 + 44)
FY = n2 + 44
shape = ffn[0] if isinstance(ffn[0], list) else ffn
widths = [d, *shape, d]
steps = [("causal conv", f"{cfg['ffn_conv_kernel']} taps")]
mid = len(shape) // 2
for i in range(len(widths) - 1):
    steps.append((f"Linear  {widths[i]} → {widths[i+1]}", None))
    if i < len(shape):
        steps.append(("centred dSiLU" if i == mid else "SiLU", None))
fh = 36 + len(steps) * 34 + 8
rect(PX - 6, FY, 450, fh, ("#fafafa", "#94a3b8"), 12, dash=True)
text(PX + 6, FY + 22, "Feed-forward", 14, "bold", "start")
text(PX + 438, FY + 22, f"{len(widths) - 1} matrices instead of 2", 12, "normal", "end", MUTED)
for j, (label, sub) in enumerate(steps):
    y = FY + 36 + j * 34
    act = "dSiLU" in label or label == "SiLU"
    box(RCX - 100, y, 200, 28, [label + (f"  ({sub})" if sub else "")], KIND["D4"] if act else PLAIN, 12.5)
    if j:
        arrow(RCX, y - 6, RCX, y)
ffn_end = FY + fh
arrow(RCX, ffn_end, RCX, ffn_end + 18)
plus(RCX, ffn_end + 29)
path(f"M{rail_x} {att_end+29}V{ffn_end+29}H{RCX+11}", MUTED, dash=True)
arrow(RCX, ffn_end + 40, RCX, ffn_end + 62)
text(RCX, ffn_end + 80, "next layer", 12.5, fill=MUTED)
right_end = ffn_end + 86

# ---- bottom: attention patterns ----------------------------------------------------------------
PY = max(left_end, right_end) + 34
rect(24, PY, W - 48, 4 * 34 + 74, ("#fafafa", "#cbd5e1"), 12)
text(40, PY + 26, "What the newest token can attend to", 14, "bold", "start")
text(W - 40, PY + 26, "schematic, not to scale: windows are shrunk 256×, the stride is kept", 11.5, "normal", "end", MUTED)
N, CW, CG = 64, 8, 1
X0 = 110
for i, kind in enumerate(pattern[:0] or ["S", "D4", "D8", "G"]):
    if kind not in types:
        continue
    t = types[kind]
    y = PY + 46 + i * 34
    text(60, y + 14, kind, 14, "bold", "middle", KIND[kind][1])
    win = None if t["window"] is None else max(2, t["window"] // 256)
    seen = set()
    for step in range(N):
        pos = N - 1 - step * t["dilation"]
        if pos < 0 or (win is not None and step >= win):
            break
        seen.add(pos)
    for c in range(N):
        fill = KIND[kind][1] if c in seen else GRID
        out.append(f'<rect x="{X0 + c * (CW + CG)}" y="{y}" width="{CW}" height="22" rx="2" fill="{fill}"/>')
    r = reach(kind)
    label = (f"window {k(t['window'])}" + (f", stride {t['dilation']}, reach {k(r)}" if t["dilation"] > 1 else f", reach {k(r)}")) if t["window"] else "every earlier token"
    text(X0 + N * (CW + CG) + 18, y + 16, label, 12.5, "normal", "start", INK)
text(X0, PY + 46 + 4 * 34 + 8, "older tokens", 11.5, "normal", "start", MUTED)
text(X0 + N * (CW + CG), PY + 46 + 4 * 34 + 8, "newest token", 11.5, "normal", "end", MUTED)

assert PY + 4 * 34 + 74 <= H, (PY + 4 * 34 + 74, H)
out.append("</svg>")
(ROOT / "docs" / "structure.svg").write_text("\n".join(out) + "\n")
print("wrote docs/structure.svg;", "layout needs", PY + 4 * 34 + 74, "of", H, "px high")
for p in problems:
    print("WARNING:", p)
