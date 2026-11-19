"""
Faithfulness, meat share and randomisation check of the explanations (Section 4.3 of the paper).

(a) deletion and (b) insertion area of each method and of a random pixel order, by predicted class, as means with
95% bootstrap intervals over specimen groups; (c) meat share minus mask area per image; (d) Spearman correlation
between the original map and the map after cascading randomisation of the weights, from the classifier down to
Stage 1. Reads the outputs of src.evaluate_xai (xai_per_image.csv, sanity_check.csv).

Usage:
    python -m src.viz.xai_measures_figure --xai runs/xai --out figures/fig_xai_measures.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})

INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
# Random order in grey; methods in Okabe-Ito blue, vermilion, bluish green
METHODS = (("random", "Random order", "#8C8C8C", "D"), ("gradcam", "Grad-CAM", "#0072B2", "o"),
           ("occlusion", "Occlusion", "#D55E00", "s"), ("shap", "SHAP", "#009E73", "^"))
CLASSES = ((0, "Fresh"), (1, "Spoiled"))
LAYERS = (("classifier", "Classifier"), ("stage4", "Stage 4"), ("s3_attn", "Stage 3\nattention"),
          ("s3_proj", "Stage 3\nprojection"), ("stage2", "Stage 2"), ("stage1", "Stage 1"))


def group_bootstrap(values, groups, n=2000, seed=0):
    """Mean and 95% interval, resampling specimen groups (as for the test metrics)."""
    rng = np.random.default_rng(seed)
    by_group = [values[groups == g] for g in np.unique(groups)]
    means = [np.concatenate([by_group[i] for i in rng.integers(len(by_group), size=len(by_group))]).mean()
             for _ in range(n)]
    return values.mean(), *np.percentile(means, [2.5, 97.5])


def long_table(per_image):
    """One row per image, predicted class and method (random order as its own method)."""
    rows = per_image[["path", "session", "pred", "method", "deletion_auc", "insertion_auc"]]
    rand = (per_image.drop_duplicates("path")[["path", "session", "pred", "random_deletion_auc",
                                                "random_insertion_auc"]]
            .rename(columns={"random_deletion_auc": "deletion_auc", "random_insertion_auc": "insertion_auc"})
            .assign(method="random"))
    return pd.concat([rand, rows], ignore_index=True)


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, width=0.6, length=2.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xai", default="runs/xai")
    ap.add_argument("--out", default="figures/fig_xai_measures.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    xai = Path(args.xai)
    per_image = pd.read_csv(xai / "xai_per_image.csv")
    sanity = pd.read_csv(xai / "sanity_check.csv")
    table = long_table(per_image)

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6))
    width = 0.18
    for ax, col, title in ((axes[0, 0], "deletion_auc", "(a) Deletion area (lower is better)"),
                           (axes[0, 1], "insertion_auc", "(b) Insertion area (higher is better)")):
        for k, (key, name, colour, marker) in enumerate(METHODS):
            for x, (cls, _) in enumerate(CLASSES):
                part = table[(table["method"] == key) & (table["pred"] == cls)]
                mean, lo, hi = group_bootstrap(part[col].to_numpy(), part["session"].to_numpy())
                pos = x + (k - 1.5) * width
                ax.errorbar(pos, mean, yerr=[[mean - lo], [hi - mean]], fmt=marker, color=colour, ms=4,
                            mfc="white" if key == "random" else colour, mew=0.9, elinewidth=0.9, capsize=1.5,
                            label=name if x == 0 else None)
        ax.set_xticks(range(len(CLASSES)), [f"Predicted {c.lower()}" for _, c in CLASSES])
        ax.set_xlim(-0.5, len(CLASSES) - 0.5)
        ax.set_ylim(0.2, 1.0)
        ax.set_title(title, loc="left", color=INK, fontsize=7)
        style(ax)
    axes[0, 0].set_ylabel("Area under the curve", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center", ncol=4, handletextpad=0.3,
               columnspacing=1.5, bbox_to_anchor=(0.5, 1.0))

    # (c) meat share minus mask area per image
    ax = axes[1, 0]
    rng = np.random.default_rng(0)
    for k, (key, name, colour, marker) in enumerate(METHODS[1:]):
        for x, (cls, _) in enumerate(CLASSES):
            g = per_image[(per_image["method"] == key) & (per_image["pred"] == cls)]
            diff = (g["meat_share"] - g["mask_area"]).to_numpy()
            pos = x + (k - 1) * 0.26
            ax.scatter(pos + rng.uniform(-0.07, 0.07, len(diff)), diff, s=3, color=colour, alpha=0.35, lw=0)
            ax.boxplot(diff, positions=[pos], widths=0.16, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="none", edgecolor=colour, linewidth=0.8),
                       medianprops=dict(color=colour, linewidth=1.2), whiskerprops=dict(color=colour, lw=0.8),
                       capprops=dict(color=colour, lw=0.8))
    ax.axhline(0, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks(range(len(CLASSES)), [f"Predicted {c.lower()}" for _, c in CLASSES])
    ax.set_xlim(-0.5, len(CLASSES) - 0.5)
    ax.set_ylabel("Meat share − mask area", color=INK)
    ax.set_title("(c) Attribution on the meat relative to its area", loc="left", color=INK, fontsize=7)
    style(ax)

    # (d) cascading randomisation
    ax = axes[1, 1]
    for key, name, colour, marker in METHODS[1:3]:
        rows = [sanity[(sanity["method"] == key) & (sanity["randomised_up_to"] == l)]["spearman"].dropna()
                for l, _ in LAYERS]
        means = [r.mean() for r in rows]
        sems = [r.std(ddof=1) / np.sqrt(len(r)) * 1.96 for r in rows]
        ax.errorbar(range(len(LAYERS)), means, yerr=sems, fmt=f"-{marker}", color=colour, ms=3.5, lw=1,
                    elinewidth=0.8, capsize=1.5, label=name)
    ax.axhline(0, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks(range(len(LAYERS)), [n for _, n in LAYERS], fontsize=6)
    ax.set_ylim(-0.4, 0.8)
    ax.set_xlabel("Weights randomised from the classifier down to", color=INK)
    ax.set_ylabel("Spearman ρ with original map", color=INK)
    ax.set_title("(d) Randomisation check", loc="left", color=INK, fontsize=7)
    style(ax)

    fig.tight_layout(h_pad=1.2, w_pad=1.5, rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=args.dpi)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
