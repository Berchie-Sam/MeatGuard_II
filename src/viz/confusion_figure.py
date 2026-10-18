"""
Confusion matrices of selected models on the test partition, from runs/results/test_results.json.

One 2 x 2 panel per model (rows = true class, columns = predicted class). Each cell gives the image count and
the share of its true class; the shading encodes that share on one scale shared by all panels, so the panels can
be compared directly.

Usage:
    python -m src.viz.confusion_figure --results runs/results/test_results.json --out figures/fig6_confusion.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})

# Panels in reading order: the strongest pretrained baseline, then MeatGuard-S through each compression step
MODELS = (("baseline:efficientnet_b0", "EfficientNet-B0"),
          ("meatguard_s:distilled", "MeatGuard-S, distilled"),
          ("meatguard_s:pruned", "MeatGuard-S, pruned\n(32-bit float)"),
          ("meatguard_s:int8", "MeatGuard-S,\nfull-integer 8-bit"),
          ("meatguard_s:deployed", "MeatGuard-S, deployed\n(8-bit, LayerNorm in float)"))
CLASSES = ("Fresh", "Spoiled")
INK, MUTED = "#222222", "#666666"
# Sequential, one hue (Okabe-Ito blue), light -> dark
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#F2F7FB", "#0072B2", "#003A5C"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="runs/results/test_results.json")
    ap.add_argument("--out", default="figures/fig6_confusion.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text())["models"]
    models = [(k, label) for k, label in MODELS if k in res]
    missing = [k for k, _ in MODELS if k not in res]
    if missing:
        print(f"not in {args.results}: {missing}")

    fig, axes = plt.subplots(1, len(models), figsize=(7.48, 1.95))
    for ax, (key, label) in zip(np.atleast_1d(axes), models):
        cm = np.array(res[key]["confusion"])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=CMAP, vmin=0, vmax=1)
        for (i, j), n in np.ndenumerate(cm):
            colour = "white" if share[i, j] > 0.55 else INK
            ax.text(j, i - 0.08, f"{n}", ha="center", va="center", color=colour, fontsize=8, fontweight="bold")
            ax.text(j, i + 0.22, f"{100 * share[i, j]:.1f}%", ha="center", va="center", color=colour, fontsize=6)
        # 2px-style surface gap between cells
        for v in (0.5,):
            ax.axhline(v, color="white", lw=1.5)
            ax.axvline(v, color="white", lw=1.5)
        ax.set_xticks([0, 1], CLASSES)
        ax.set_yticks([0, 1], CLASSES, rotation=90, va="center")
        ax.tick_params(length=0, pad=2, colors=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, fontsize=7, color=INK, pad=4)
        ax.set_xlabel("Predicted", color=MUTED, labelpad=2)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("True", color=MUTED, labelpad=2)

    fig.subplots_adjust(left=0.035, right=0.995, top=0.80, bottom=0.14, wspace=0.28)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
