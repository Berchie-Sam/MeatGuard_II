"""
Validation curves of all trained models (Appendix A of the paper).

Validation macro-F1 (top row) and validation loss (bottom row) per epoch, read from the history.csv of each run:
(a) teacher candidates, (b) pretrained baselines and MeatGuard-S without and with distillation, (c) pruned
MeatGuard-S during fine-tuning, starting from the model right after pruning (epoch 0, from the pruning summary) with
the unpruned model as a reference. The checkpoint kept for each run (summary.json, best_epoch) is marked. Only
validation data are shown; epochs are counted from 1. EfficientNet-B0 was trained once and serves both as a teacher
candidate and as a baseline, so it is drawn in (a) only.

Usage:
    python -m src.viz.training_curves_figure --runs runs --out figures/figA1_training_curves.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})

INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
# Okabe-Ito colours
PANELS = (
    ("(a) Teacher candidates", (("teacher/resnet50", "ResNet-50 (selected)", "#000000"),
                                ("teacher/swin_t", "Swin-T", "#999999"),
                                ("teacher/convnext_tiny", "ConvNeXt-T", "#E69F00"),
                                ("teacher/efficientnet_b0", "EfficientNet-B0 (also a baseline)", "#CC79A7"))),
    ("(b) Baselines and MeatGuard-S", (("baselines/mobilenet_v3_small", "MobileNetV3-Small", "#56B4E9"),
                                       ("baselines/shufflenet_v2_x1_0", "ShuffleNet V2", "#E69F00"),
                                       ("student/scratch", "MeatGuard-S, no distillation", "#0072B2"),
                                       ("student/distilled", "MeatGuard-S, distilled", "#D55E00"))),
    ("(c) Pruning and fine-tuning", (("student/pruned", "Pruned, fine-tuned", "#009E73"),)),
)


def style(ax):
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, width=0.6, length=2.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures/figA1_training_curves.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    runs = Path(args.runs)

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.2), sharey="row",
                             gridspec_kw={"width_ratios": [1, 1.6, 0.62]})
    for c, (title, curves) in enumerate(PANELS):
        top, bottom = axes[0, c], axes[1, c]
        for run, name, colour in curves:
            h = pd.read_csv(runs / run / "history.csv")
            summary = json.loads((runs / run / "summary.json").read_text())
            epoch = h["epoch"] + 1
            best = h[h["epoch"] == summary["best_epoch"]]
            if run == "student/pruned":
                # Epoch 0: right after pruning, before fine-tuning
                before, ref = summary["val_after_pruning_before_finetune"], summary["val_before_pruning"]
                epoch = pd.concat([pd.Series([0]), epoch], ignore_index=True)
                f1 = pd.concat([pd.Series([before["macro_f1"]]), h["val_macro_f1"]], ignore_index=True)
                loss = pd.concat([pd.Series([before["loss"]]), h["val_loss"]], ignore_index=True)
                for ax, value in ((top, ref["macro_f1"]), (bottom, ref["loss"])):
                    ax.axhline(value, color=MUTED, lw=0.7, ls="--", label="Before pruning" if ax is top else None)
                for ax, value in ((top, before["macro_f1"]), (bottom, before["loss"])):
                    ax.plot(0, value, "x", color=colour, ms=4, mew=1, label="After pruning" if ax is top else None)
            else:
                f1, loss = h["val_macro_f1"], h["val_loss"]
            top.plot(epoch, f1, color=colour, lw=0.9, label=name)
            bottom.plot(epoch, loss, color=colour, lw=0.9)
            for ax, col in ((top, "val_macro_f1"), (bottom, "val_loss")):
                ax.plot(best["epoch"] + 1, best[col], "o", color=colour, ms=3.5, mfc="white", mew=0.9)
        top.set_title(title, loc="left", color=INK, fontsize=7)
        top.legend(frameon=False, fontsize=5.5, loc="center right" if c == 2 else "lower right", handlelength=1.4)
        bottom.set_xlabel("Epoch", color=INK)
        for ax in (top, bottom):
            style(ax)
    axes[0, 0].set_ylim(0.3, 0.95)
    axes[0, 0].set_ylabel("Validation macro-F1", color=INK)
    axes[1, 0].set_ylabel("Validation loss", color=INK)
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
