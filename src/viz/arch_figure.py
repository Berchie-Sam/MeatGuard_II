"""
Fig. 2 — MeatGuard-S architecture.

(a) Block-level network view rendered with VisualTorch (Hendria & Gavrikov, 2024). A display model with one
    opaque module per block is rendered; before rendering, its output shapes are checked against forward hooks
    on the real MeatGuardS, so the figure cannot drift from the implementation.
(b)-(d) Inset diagrams of the inverted residual block, the Stage-3 transformer block and the SE block.

Usage (needs Python >= 3.10 for VisualTorch >= 1.0, e.g. the .venv-viz environment):
    python -m src.viz.arch_figure --out figures/fig2_architecture.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import visualtorch
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import ImageFont

from src.model.architecture import MeatGuardS
from src.viz.flowcharts import save_figure

# Okabe-Ito colours, one per block type (figure_plan.md)
INPUT_COL = "#D9D9D9"
COL = {"Stem conv": "#E69F00", "Depthwise-separable conv": "#F5C26B", "Inverted residual": "#56B4E9",
       "1×1 projection": "#999999", "Max pool": "#F0E442", "Transformer block": "#CC79A7",
       "Squeeze-and-excitation": "#009E73", "Global average pool": "#0072B2", "Linear classifier": "#D55E00"}
INK = "#222222"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "mathtext.fontset": "dejavusans"})


class Block(nn.Module):
    """Opaque display block: no children, so VisualTorch draws it as one box with the true output shape."""
    def __init__(self, channels, stride=1, pool=False):
        super().__init__()
        self.channels, self.stride, self.pool = channels, stride, pool

    def forward(self, x):
        if self.pool:
            return x.mean((2, 3))
        if x.dim() == 2:
            return x[:, :self.channels]
        if self.stride > 1:
            x = F.avg_pool2d(x, self.stride)
        return x.new_zeros(x.shape[0], self.channels, x.shape[2], x.shape[3])


# One subclass per legend entry (VisualTorch colours boxes by module type)
TYPES = {name: type("".join(ch if ch.isalnum() else "_" for ch in name), (Block,), {}) for name in COL}

# (legend name, constructor args, real submodule whose output must match)
LAYOUT = [
    ("Stem conv", dict(channels=32, stride=2), lambda m: m.stage1[2]),
    ("Depthwise-separable conv", dict(channels=64, stride=2), lambda m: m.stage1),
    ("Inverted residual", dict(channels=128, stride=2), lambda m: m.stage2[0]),
    ("Inverted residual", dict(channels=128), lambda m: m.stage2[1]),
    ("Inverted residual", dict(channels=128), lambda m: m.stage2[2]),
    ("1×1 projection", dict(channels=256), lambda m: m.s3_proj),
    ("Max pool", dict(channels=256, stride=2), lambda m: m.s3_down),
    ("Transformer block", dict(channels=256), lambda m: m.s3_attn),
    ("Inverted residual", dict(channels=384), lambda m: m.stage4[0]),
    ("Squeeze-and-excitation", dict(channels=384), lambda m: m.stage4[1]),
    ("Max pool", dict(channels=384, stride=2), lambda m: m.s4_down),
    ("Global average pool", dict(channels=384, pool=True), lambda m: m.gap),
    ("Linear classifier", dict(channels=2), lambda m: m.classifier),
]


def build_display_model():
    return nn.Sequential(*[TYPES[name](**kw) for name, kw, _ in LAYOUT])


def check_against_real(display, size=224):
    """Assert every display block's output shape equals the corresponding real MeatGuardS submodule's."""
    real = MeatGuardS().eval()
    captured = []
    hooks = [getter(real).register_forward_hook(lambda mod, i, o: captured.append(tuple(o.shape[1:])))
             for _, _, getter in LAYOUT]
    x = torch.randn(1, 3, size, size)
    shown, h = [], x
    with torch.no_grad():
        real(x)
        for block in display:
            h = block(h)
            shown.append(tuple(h.shape[1:]))
    for hk in hooks:
        hk.remove()
    # global average pooling gives (C, 1, 1) in the real model and (C,) in the display model
    norm = lambda s: (s[0],) if len(s) == 3 and s[1:] == (1, 1) else s
    mismatch = [(LAYOUT[i][0], a, b) for i, (a, b) in enumerate(zip(map(norm, captured), shown)) if a != b]
    if mismatch:
        raise AssertionError(f"Display model does not match MeatGuardS: {mismatch}")
    return shown


def render_network(display, font_path, size=224):
    font = ImageFont.truetype(font_path, 23)
    cmap = {TYPES[n]: {"fill": c} for n, c in COL.items()}
    cmap[visualtorch.Input] = {"fill": INPUT_COL}
    return visualtorch.render(display, (1, 3, size, size), style="flow", color_map=cmap, legend=False,
                              show_dimension=True, spacing=56, scale_xy=1.7, scale_z=0.7, max_z=260, max_xy=270,
                              font=font, padding=30)


# ----------------------------------------------------------------------------- insets (matplotlib)
def node(ax, x, y, w, h, text, colour, fs=6.2, tint=0.18):
    rgb = np.array(to_rgb(colour))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=tuple(rgb * tint + (1 - tint)), ec=colour, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def link(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=INK,
                                 shrinkA=0, shrinkB=0))


def stack(ax, title, items, colour, skips=(), cx=0.56, box_w=0.72):
    """Operations stacked bottom to top. items: (kind, text) with kind in {io, n, op};
    skips: (source index, target op index, x of the side rail)."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=7.5, weight="bold", loc="left", pad=4)
    ys = np.linspace(0.06, 0.94, len(items))
    half = {"op": 0.035, "n": 0.045, "io": 0.022}
    for k, (kind, text) in enumerate(items):
        if kind == "op":
            ax.add_patch(plt.Circle((cx, ys[k]), 0.035, fc="white", ec=INK, lw=0.8))
            ax.text(cx, ys[k] - 0.003, text, ha="center", va="center", fontsize=8)
        elif kind == "io":
            ax.text(cx, ys[k], text, ha="center", va="center", fontsize=6.2, style="italic")
        else:
            node(ax, cx, ys[k], box_w, 0.09, text, colour)
        if k:
            link(ax, cx, ys[k - 1] + half[items[k - 1][0]], cx, ys[k] - half[kind])
    for src, dst, x_side in skips:
        x_start = cx - 0.035 if items[src][0] == "op" else cx - 0.02
        if items[src][0] == "op":          # branch off the line just above the operator
            x_start, y_start = cx, ys[src] + half["op"] + 0.03
        else:
            y_start = ys[src] + half["io"] + 0.012
        ax.plot([x_start, x_side, x_side], [y_start, y_start, ys[dst]], color=INK, lw=0.8)
        link(ax, x_side, ys[dst], cx - 0.035, ys[dst])
    return ys


def draw_b(ax):
    ys = stack(ax, "(b) Inverted residual block", [
        ("io", "input  H × W × C"),
        ("n", "1 × 1 conv, BN, SiLU\n(expand to 2C)"),
        ("n", "3 × 3 depthwise conv\n(stride s), BN, SiLU"),
        ("n", "1 × 1 conv, BN\n(linear projection)"),
        ("op", "+"),
        ("io", "output"),
    ], COL["Inverted residual"], skips=[(0, 4, 0.08)])
    ax.text(0.055, (ys[1] + ys[3]) / 2, "only if s = 1 and shapes match", ha="center", va="center",
            fontsize=5.2, color="#555555", rotation=90)


def draw_c(ax):
    stack(ax, "(c) Stage-3 transformer block", [
        ("io", "14 × 14 × 256 → 196 tokens"),
        ("n", "LayerNorm"),
        ("n", "Multi-head self-attention\n" + r"(4 heads, $d_h$ = 64)"),
        ("op", "+"),
        ("n", "LayerNorm"),
        ("n", "FFN: 1 × 1 conv 256→512,\nSiLU, 1 × 1 conv 512→256"),
        ("op", "+"),
        ("io", "14 × 14 × 256"),
    ], COL["Transformer block"], skips=[(0, 3, 0.07), (3, 6, 0.14)])


def draw_d(ax):
    stack(ax, "(d) Squeeze-and-excitation", [
        ("io", "input  14 × 14 × 384"),
        ("n", "Global average pool"),
        ("n", "FC 384→96, SiLU"),
        ("n", "FC 96→384, sigmoid"),
        ("op", "×"),
        ("io", "channel-reweighted output"),
    ], COL["Squeeze-and-excitation"], skips=[(0, 4, 0.08)])


def draw_legend(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    legend = {"Input": INPUT_COL, **COL}
    for k, name in enumerate(legend):
        r, c = divmod(k, 5)
        x, y = 0.02 + c * 0.196, 0.72 - r * 0.42
        ax.add_patch(plt.Rectangle((x, y - 0.12), 0.018, 0.24, fc=legend[name], ec="#444444", lw=0.4))
        ax.text(x + 0.025, y, name, va="center", fontsize=6.4)


def figure_a(net, path, dpi, W=7.48):
    title_h, legend_h = 0.24, 0.45
    net_h = W * net.size[1] / net.size[0]
    H = title_h + net_h + legend_h
    fig = plt.figure(figsize=(W, H))
    fig.text(0.01, 1 - 0.05 / H, "(a) MeatGuard-S, block level (output size per block)", fontsize=7.5,
             weight="bold", va="top")
    ax = fig.add_axes([0.0, legend_h / H, 1.0, net_h / H])
    ax.imshow(np.asarray(net)); ax.axis("off")
    draw_legend(fig.add_axes([0.0, 0.0, 1.0, legend_h / H]))
    save_figure(fig, path, dpi=dpi); plt.close(fig); print("wrote", path)


def figure_inset(draw, path, dpi, size=(2.5, 3.1)):
    fig = plt.figure(figsize=size)
    draw(fig.add_axes([0.03, 0.02, 0.94, 0.9]))
    save_figure(fig, path, dpi=dpi); plt.close(fig); print("wrote", path)


# ----------------------------------------------------------------------------- interaction figure
def figure_interaction(path, dpi):
    """Where blocks (b)-(d) sit in MeatGuard-S, and what connects to them during training and inference."""
    W, H = 7.48, 4.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    TEACH, LOSS, GRAD, XAI = "#7F7F7F", "#E69F00", "#D55E00", "#0072B2"
    DASH, DOT = (0, (3, 2)), (0, (1, 1.5))

    def tint(c, t=0.16):
        return tuple(np.array(to_rgb(c)) * t + (1 - t))

    def rbox(x, y, w, h, text, ec, fc="white", fs=6.6, bold=False, z=2, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec,
                                    lw=1.0, ls=ls, zorder=z))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=z + 1,
                    weight="bold" if bold else "normal", linespacing=1.2)

    def arr(pts, colour=INK, ls="-", lw=0.9, z=4):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z + 1,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, mutation_scale=7, shrinkA=0, shrinkB=0))

    def badge(x, y, letter, colour):
        ax.add_patch(plt.Circle((x, y), 0.105, fc=colour, ec="white", lw=0.8, zorder=6))
        ax.text(x, y - 0.005, letter, ha="center", va="center", fontsize=6.4, weight="bold", color="white", zorder=7)

    # --- student forward pass (middle row)
    y0, h = 1.95, 0.78
    ax.text(0.12, y0 + h + 0.2, "Student: MeatGuard-S", fontsize=7.5, weight="bold", color="#B37700")
    stages = [  # x, width, title, body, colour, badges of the inset blocks used inside
        (0.95, 0.95, "Stage 1", "Stem conv +\ndepthwise-sep.", COL["Stem conv"], []),
        (2.08, 0.95, "Stage 2", "3 × inverted\nresidual", COL["Inverted residual"], [("b", COL["Inverted residual"])]),
        (3.21, 1.1, "Stage 3", "1 × 1 proj., pool,\ntransformer", COL["Transformer block"],
         [("c", COL["Transformer block"])]),
        (4.49, 1.05, "Stage 4", "Inv. residual\n+ SE, pool", COL["Squeeze-and-excitation"],
         [("d", COL["Squeeze-and-excitation"]), ("b", COL["Inverted residual"])]),
        (5.72, 0.62, "Head", "GAP,\nFC", COL["Linear classifier"], []),
    ]
    rbox(0.12, y0 + 0.14, 0.62, 0.5, "Crop\n224²", "#999999", fc="#F2F2F2", fs=6.2)
    sizes = ["", "56²×64", "28²×128", "14²×256", "7²×384"]
    prev_x = 0.74
    for k, (x, w, title, body, col, badges) in enumerate(stages):
        rbox(x, y0, w, h, "", col, fc=tint(col))
        ax.text(x + w / 2, y0 + h - 0.16, title, ha="center", va="center", fontsize=6.8, weight="bold", zorder=4)
        ax.text(x + w / 2, y0 + 0.3, body, ha="center", va="center", fontsize=6.0, zorder=4, linespacing=1.15)
        for j, (letter, bcol) in enumerate(badges):
            badge(x + w - 0.13 - j * 0.24, y0 + h, letter, bcol)
        arr([(prev_x + 0.02, y0 + h / 2), (x - 0.02, y0 + h / 2)])
        if sizes[k]:
            ax.text((prev_x + x) / 2, y0 + h / 2 + 0.06, sizes[k], ha="center", va="bottom", fontsize=5.2,
                    color="#555555", zorder=7, bbox=dict(fc="white", ec="none", pad=0.6))
        prev_x = x + w
    rbox(6.62, y0 + 0.2, 0.62, 0.38, r"logits $z_s$", COL["Linear classifier"], fs=6.2)
    arr([(prev_x + 0.02, y0 + h / 2), (6.6, y0 + h / 2)])

    # --- teacher (top row)
    yt = 3.55
    ax.text(0.12, yt + 0.62, "Teacher (frozen: no gradient)", fontsize=7.5, weight="bold", color=TEACH)
    rbox(0.12, yt + 0.06, 0.62, 0.42, "Crop", "#999999", fc="#F2F2F2", fs=6.2)
    rbox(1.0, yt, 3.0, 0.55, "ImageNet-pretrained CNN/ViT, selected on validation", TEACH, fc=tint(TEACH), fs=6.4)
    arr([(0.76, yt + 0.27), (0.98, yt + 0.27)], colour=TEACH)
    rbox(4.35, yt + 0.06, 0.92, 0.42, r"features $\varphi_t$", TEACH, fs=6.2)
    rbox(6.62, yt + 0.06, 0.62, 0.42, r"logits $z_t$", TEACH, fs=6.2)
    arr([(4.02, yt + 0.27), (4.33, yt + 0.27)], colour=TEACH)
    arr([(5.29, yt + 0.27), (6.6, yt + 0.27)], colour=TEACH)
    ax.text(5.95, yt + 0.32, "classifier", ha="center", va="bottom", fontsize=5.4, color=TEACH)

    # --- losses (bottom row)
    yl = 0.62
    rbox(3.0, yl, 1.75, 0.46, r"Alignment  $\Vert A(\bar\varphi_s)-\varphi_t\Vert^2$", LOSS, fc=tint(LOSS), fs=6.0)
    rbox(4.95, yl, 1.15, 0.46, r"KL$(\sigma(z_t/T)\,\Vert\,\sigma(z_s/T))$", LOSS, fc=tint(LOSS), fs=5.6)
    rbox(6.3, yl, 0.95, 0.46, r"CE$(y, z_s)$", LOSS, fc=tint(LOSS), fs=6.0)
    rbox(3.0, 0.06, 4.25, 0.34, r"$\mathcal{L}=\alpha\,\mathrm{CE}+(1-\alpha)\,T^{2}\,\mathrm{KL}+\beta\cdot\mathrm{alignment}$",
         LOSS, fs=6.4, bold=True)
    for x in (3.87, 5.52, 6.77):
        arr([(x, yl - 0.005), (x, 0.42)], colour=LOSS, lw=0.8)

    # Stage-3 output -> GAP -> adaptor A -> alignment
    x3 = 3.21 + 0.55
    rbox(2.72, 1.27, 0.72, 0.4, "GAP of\nStage-3 output", COL["Global average pool"], fs=5.6)
    rbox(3.53, 1.3, 0.77, 0.34, r"adaptor $A$", LOSS, fs=6.0)
    arr([(x3, y0 - 0.01), (x3, 1.76), (3.08, 1.76), (3.08, 1.69)], colour=COL["Transformer block"])
    arr([(3.46, 1.47), (3.51, 1.47)])
    arr([(3.9, 1.28), (3.9, yl + 0.48)], colour=LOSS)
    # teacher features -> alignment, through the gap between Stage 3 and Stage 4
    xg = 4.40
    arr([(4.81, yt + 0.04), (4.81, 3.3), (xg, 3.3), (xg, yl + 0.48)], colour=TEACH)
    # teacher logits -> KL, through the gap between the head and the student logits
    xg2 = 6.48
    arr([(6.93, yt + 0.04), (6.93, 3.3), (xg2, 3.3), (xg2, 1.22), (5.85, 1.22), (5.85, yl + 0.48)], colour=TEACH)
    # student logits -> KL and CE; labels -> CE
    arr([(6.75, y0 + 0.18), (6.75, 1.45), (5.3, 1.45), (5.3, yl + 0.48)])
    arr([(7.05, y0 + 0.18), (7.05, yl + 0.48)])
    ax.text(7.32, 1.3, "+ labels y", rotation=90, ha="center", va="center", fontsize=5.6, color="#555555")

    # gradients (training): from the total loss back into every student stage
    ybar = 1.86
    arr([(3.0, 0.23), (2.62, 0.23), (2.62, ybar)], colour=GRAD, ls=DASH, lw=1.0, z=3)
    ax.plot([1.42, 5.0], [ybar, ybar], color=GRAD, lw=1.0, ls=DASH, zorder=3)
    for x in (1.42, 2.55, 3.45, 5.0):
        arr([(x, ybar), (x, y0 - 0.02)], colour=GRAD, ls=DASH, lw=1.0, z=3)
    ax.text(1.1, 1.35, "gradients reach all student\nstages and the adaptor;\nthe teacher stays frozen",
            fontsize=5.6, color=GRAD, ha="left", va="center", linespacing=1.15)

    # Grad-CAM (inference) reads Stage-4 activations and gradients
    rbox(4.75, 2.98, 0.82, 0.3, "Grad-CAM", XAI, fs=6.0, ls=DOT)
    arr([(5.16, y0 + h + 0.12), (5.16, 2.96)], colour=XAI, ls=DOT, lw=1.0)

    # key
    kx, ky = 0.12, 1.0
    for k, (label, colour, ls) in enumerate([("forward", INK, "-"), ("teacher (frozen)", TEACH, "-"),
                                              ("gradient (training)", GRAD, DASH),
                                              ("explanation (inference)", XAI, DOT)]):
        yy = ky - k * 0.18
        ax.plot([kx, kx + 0.3], [yy, yy], color=colour, ls=ls, lw=1.0)
        ax.text(kx + 0.36, yy, label, va="center", fontsize=5.8)
    ax.text(0.12, 0.16, "Badges b, c, d mark the blocks\ndetailed in panels (b)-(d)", fontsize=5.6, color="#555555",
            va="center", linespacing=1.2)

    save_figure(fig, path, dpi=dpi); plt.close(fig); print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="combined figure (a)-(d)")
    ap.add_argument("--parts-dir", default=None,
                    help="also save (a), (b), (c), (d) and the block-interaction figure as separate images")
    ap.add_argument("--font", default="C:/Windows/Fonts/arial.ttf")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    display = build_display_model()
    shapes = check_against_real(display)
    print("display shapes verified against MeatGuardS:", shapes)
    net = render_network(display, args.font)

    W = 7.48                                   # inches, Elsevier full width
    title_h, legend_h, inset_h = 0.24, 0.42, 2.75
    net_h = W * net.size[1] / net.size[0]
    H = title_h + net_h + legend_h + inset_h
    fig = plt.figure(figsize=(W, H))
    fig.text(0.01, 1 - 0.05 / H, "(a) MeatGuard-S, block level (output size per block)", fontsize=7.5,
             weight="bold", va="top")
    ax_a = fig.add_axes([0.0, 1 - (title_h + net_h) / H, 1.0, net_h / H])
    ax_a.imshow(np.asarray(net)); ax_a.axis("off")
    draw_legend(fig.add_axes([0.0, inset_h / H, 1.0, legend_h / H]))
    ins = (inset_h - 0.3) / H
    for draw, rect in ((draw_b, [0.02, 0.02, 0.3, ins]), (draw_c, [0.345, 0.02, 0.31, ins]),
                       (draw_d, [0.68, 0.02, 0.3, ins])):
        draw(fig.add_axes(rect))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.out, dpi=args.dpi); plt.close(fig)
    print("wrote", args.out)

    if args.parts_dir:
        out = Path(args.parts_dir); out.mkdir(parents=True, exist_ok=True)
        figure_a(net, out / "fig2a_network.png", args.dpi)
        figure_inset(draw_b, out / "fig2b_inverted_residual.png", args.dpi)
        figure_inset(draw_c, out / "fig2c_transformer_block.png", args.dpi, size=(2.6, 3.4))
        figure_inset(draw_d, out / "fig2d_squeeze_excitation.png", args.dpi)
        figure_interaction(out / "fig2e_block_interactions.png", args.dpi)


if __name__ == "__main__":
    main()
