"""
Fig. 1 — MeatGuard II experiment pipeline (data -> protocol -> training -> compression/deployment -> evaluation).

Uses real MeatScan images: one raw spoiled image with its detector box, SAM mask and crop, and a run of
consecutive frames bracketed by capture session and labelled with their partition (from sessions.csv).

Usage:
    python -m src.viz.pipeline_figure --raw data/raw/MeatScan --full data/processed/full \
        --out figures/fig1_pipeline.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox
from PIL import Image

# Okabe-Ito palette (figure_plan.md): fixed colour per concept across all figures
C = {"data": "#0072B2", "teacher": "#7F7F7F", "student": "#E69F00", "attn": "#CC79A7",
     "compress": "#009E73", "deploy": "#D55E00", "eval": "#56B4E9", "ink": "#222222", "soft": "#F4F4F4"}
SPLIT_COLOUR = {"train": "#0072B2", "val": "#E69F00", "test": "#D55E00"}
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})

W, H = 7.48, 4.15                          # inches; Elsevier full width (190 mm)
TOP = 0.37                                 # top panels span TOP..0.985; evaluation band below


def box(ax, x, y, w, h, text, colour, fc=None, fs=7, weight="normal", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc or "white", ec=colour, lw=lw, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight,
                color=C["ink"], zorder=3, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, colour=C["ink"], lw=0.9, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, lw=lw, color=colour,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0))


def image(fig, rect, img, border=None):
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border or "#BBBBBB"); s.set_linewidth(1.6 if border else 0.5)
    return ax


def panel(ax, x, w, num, title, colour):
    ax.add_patch(Rectangle((x, TOP), w, 0.985 - TOP, fc=C["soft"], ec="none", zorder=0))
    ax.text(x + 0.01, 0.948, f"{num}  {title}", ha="left", va="bottom", fontsize=8, weight="bold", color=colour)


def draw_data_panel(fig, ax, raw, full, example, frames, first_frame, n_frames):
    panel(ax, 0.008, 0.432, "1", "Data and leakage-controlled protocol", C["data"])
    stem = Path(example).with_suffix("")
    seg = pd.read_csv(full / "segmentation.csv").set_index("path")
    raw_img = np.asarray(Image.open(raw / example).convert("RGB"))
    mask = np.asarray(Image.open(full / "masks" / f"{stem}.png")) > 127
    crop = np.asarray(Image.open(full / "crops" / f"{stem}.jpg").convert("RGB"))
    boxes = [list(map(float, b.split(","))) for b in seg.loc[example, "boxes"].split(";")]

    s, y_img = 0.088, 0.705                # thumbnail width (fraction of figure width) and bottom edge
    sh = s * W / H
    xs = [0.025, 0.128, 0.231, 0.334]
    image(fig, [xs[0], y_img, s, sh], raw_img)
    a1 = image(fig, [xs[1], y_img, s, sh], raw_img)
    for x0, y0, x1, y1 in boxes:
        a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#F0E442", lw=1.4))
    overlay = raw_img.astype(float)
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([0, 158, 115])
    image(fig, [xs[2], y_img, s, sh], overlay.astype(np.uint8))
    image(fig, [xs[3], y_img, s, sh], crop, border=C["student"])
    captions = ["MeatScan image\n(smartphone, market)", "Grounding DINO\n\"raw meat.\" box",
                "SAM 2.1 mask", "Cropped input\n224 × 224"]
    for x, cap in zip(xs, captions):
        ax.text(x + s / 2, y_img - 0.012, cap, ha="center", va="top", fontsize=6.2, linespacing=1.15)
    for i in range(3):
        arrow(ax, xs[i] + s + 0.003, y_img + sh / 2, xs[i + 1] - 0.003, y_img + sh / 2)

    # Session strip: consecutive frames; one bracket per capture session, labelled with its partition
    sessions = pd.read_csv(full / "sessions.csv")
    sessions["n"] = sessions["path"].str.extract(r"\((\d+)\)", expand=False).astype(int)
    strip = sessions[(sessions["class"] == frames) & (sessions["n"] >= first_frame)] \
        .sort_values("n").head(n_frames).reset_index(drop=True)
    fs_, gap = 0.04, 0.005
    fsh = fs_ * W / H
    x_strip, y_strip = 0.025, 0.462
    ax.text(x_strip, y_strip + fsh + 0.012, "Consecutive frames linked into capture sessions by\n"
            "perceptual hashing; whole sessions go to one partition", ha="left", va="bottom", fontsize=6.2,
            linespacing=1.2)
    for k, r in strip.iterrows():
        image(fig, [x_strip + k * (fs_ + gap), y_strip, fs_, fsh], Image.open(raw / r["path"]).convert("RGB"))
    run_start = 0
    for k in range(1, len(strip) + 1):
        if k == len(strip) or strip.loc[k, "session"] != strip.loc[run_start, "session"]:
            x0 = x_strip + run_start * (fs_ + gap) + 0.002
            x1 = x_strip + (k - 1) * (fs_ + gap) + fs_ - 0.002
            split = strip.loc[run_start, "split"]
            col = SPLIT_COLOUR[split]
            ax.plot([x0, x0, x1, x1], [y_strip - 0.006, y_strip - 0.018, y_strip - 0.018, y_strip - 0.006],
                    color=col, lw=1.1, solid_capstyle="butt")
            ax.text((x0 + x1) / 2, y_strip - 0.024, split, ha="center", va="top", fontsize=6, color=col,
                    weight="bold")
            run_start = k
    x_end = x_strip + len(strip) * (fs_ + gap)
    box(ax, x_end + 0.008, 0.405, 0.43 - x_end - 0.004, 0.13,
        "Session split\n7,362 / 1,578 / 1,578\nimages; no session\ncrosses partitions", C["data"], fs=6.1)
    return crop, (xs[3] + s, y_img + sh * 0.35)


def draw_training_panel(ax, crop_anchor, connect=True):
    panel(ax, 0.452, 0.255, "2", "Knowledge distillation", C["student"])
    box(ax, 0.47, 0.76, 0.22, 0.115, "Teacher: ImageNet-pretrained CNN/ViT\nfine-tuned on training sessions,\n"
        "selected on validation macro-F1", C["teacher"], fc="#EFEFEF", fs=6.2)
    box(ax, 0.47, 0.475, 0.22, 0.13, "MeatGuard-S (student)\nCNN–Transformer hybrid\n1.14 M params · 0.38 GMACs",
        C["student"], fc="#FCEFD9", fs=6.4, weight="bold")
    arrow(ax, 0.58, 0.76, 0.58, 0.607, C["teacher"])
    ax.text(0.588, 0.683, "soft labels (T = 4)\n+ feature alignment", ha="left", va="center", fontsize=6,
            color=C["teacher"], linespacing=1.2)
    ax.text(0.58, 0.415, r"$\mathcal{L}=\alpha\,\mathrm{CE}+(1-\alpha)\,T^{2}\,\mathrm{KL}+\beta\,"
            r"\Vert A(\varphi_s)-\varphi_t\Vert^{2}$", ha="center", va="center", fontsize=6.8, color=C["ink"])
    if connect:
        arrow(ax, crop_anchor[0] + 0.004, crop_anchor[1], 0.468, 0.56, C["data"], rad=0.25)


def draw_deploy_panel(fig, ax, crop, connect=True):
    panel(ax, 0.718, 0.274, "3", "Compression and deployment", C["compress"])
    steps = [("Structured pruning\n(L1 channels)", 0.79), ("Fine-tuning", 0.675), ("INT8 post-training\nquantisation", 0.56)]
    bx, bw, bh = 0.735, 0.14, 0.08
    for text, y in steps:
        box(ax, bx, y, bw, bh, text, C["compress"], fs=6.1)
    for (_, y0), (_, y1) in zip(steps[:-1], steps[1:]):
        arrow(ax, bx + bw / 2, y0, bx + bw / 2, y1 + bh, C["compress"])
    ax.text(bx + bw / 2, 0.548, "calibration: training split", ha="center", va="top", fontsize=5.6, color=C["compress"])
    if connect:
        arrow(ax, 0.692, 0.56, bx - 0.002, 0.83, C["student"], rad=-0.3)
    px, py, pw, ph = 0.895, 0.46, 0.085, 0.43      # phone outline
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0.002,rounding_size=0.015", fc="white",
                                ec=C["deploy"], lw=1.1, zorder=2))
    image(fig, [px + 0.009, py + 0.2, pw - 0.018, (pw - 0.018) * W / H], crop)
    ax.text(px + pw / 2, py + 0.155, "Fresh / Spoiled\n+ confidence", ha="center", va="center", fontsize=5.8,
            weight="bold", color=C["deploy"], linespacing=1.2)
    ax.text(px + pw / 2, py + 0.07, "Grad-CAM\noverlay", ha="center", va="center", fontsize=5.6, linespacing=1.2)
    ax.text(px + pw / 2, py - 0.012, "Offline Android\napp (TFLite)", ha="center", va="top", fontsize=6, linespacing=1.15)
    arrow(ax, bx + bw + 0.002, 0.6, px - 0.002, 0.67, C["compress"])


def draw_evaluation_band(ax, connect=True):
    ax.add_patch(Rectangle((0.008, 0.02), 0.984, 0.325, fc="white", ec="#DDDDDD", lw=0.6, zorder=0))
    ax.text(0.018, 0.305, "4  Evaluation on the held-out test sessions", ha="left", va="bottom", fontsize=8,
            weight="bold", color=C["eval"])
    cols = [
        ("Classification", "Accuracy, macro-F1,\nspoiled-class recall, ROC-AUC;\n95% CIs by resampling sessions"),
        ("Baselines", "Teacher; MobileNetV3-Small,\nEfficientNet-B0, ShuffleNetV2\n(same split and crops)"),
        ("Efficiency", "Parameters, MACs, model size,\nlatency measured on\nAndroid smartphones"),
        ("Explanations", "Grad-CAM, SHAP, occlusion;\ndeletion / insertion AUC,\nmeat share, randomisation check"),
    ]
    cw = 0.225
    for k, (title, body) in enumerate(cols):
        x = 0.025 + k * (cw + 0.018)
        box(ax, x, 0.045, cw, 0.215, "", C["eval"])
        ax.text(x + cw / 2, 0.232, title, ha="center", va="center", fontsize=7, weight="bold", color=C["eval"])
        ax.text(x + cw / 2, 0.135, body, ha="center", va="center", fontsize=6.3, linespacing=1.35)
    for x in ((0.224, 0.58, 0.855) if connect else ()):
        arrow(ax, x, TOP, x, 0.345, C["eval"], style="-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/MeatScan")
    ap.add_argument("--full", default="data/processed/full")
    ap.add_argument("--example", default="Spoiled_CowMeat/Spoiled (1000).jpg")
    ap.add_argument("--frames", default="Fresh_CowMeat", help="class folder for the session strip")
    ap.add_argument("--first-frame", type=int, default=470)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--panels-dir", default=None, help="also save each stage as a standalone image here")
    args = ap.parse_args()

    def build(connect):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        crop, crop_anchor = draw_data_panel(fig, ax, Path(args.raw), Path(args.full), args.example,
                                            args.frames, args.first_frame, args.n_frames)
        draw_training_panel(ax, crop_anchor, connect)
        draw_deploy_panel(fig, ax, crop, connect)
        draw_evaluation_band(ax, connect)
        return fig

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig = build(connect=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print("wrote", args.out)

    if args.panels_dir:
        # Each stage as its own image: same drawing without the arrows that cross panels, cropped to the panel
        out_dir = Path(args.panels_dir); out_dir.mkdir(parents=True, exist_ok=True)
        panels = {"stage1_data_protocol": (0.008, TOP, 0.44, 0.985),
                  "stage2_distillation": (0.452, TOP, 0.707, 0.985),
                  "stage3_compression_deployment": (0.718, TOP, 0.992, 0.985),
                  "stage4_evaluation": (0.008, 0.02, 0.992, 0.345)}
        fig = build(connect=False)
        for name, (x0, y0, x1, y1) in panels.items():
            bbox = Bbox([[x0 * W, y0 * H], [x1 * W, y1 * H]])
            path = out_dir / f"fig1_{name}.png"
            fig.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches=bbox)
            print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()
