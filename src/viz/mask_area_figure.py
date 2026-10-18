"""
Spoiled-class recall and the extent of the meat region (Section 4.2 of the paper).

Mask area = fraction of the original frame covered by the meat mask (segmentation.csv of src.data.segment_meat).
(a) spoiled-class recall of each model within quartiles of mask area over the spoiled test images; (b) distribution
of mask area for fresh and spoiled test images. Also writes the statistics quoted in the text (ROC AUC of the mask
area alone, per-quartile recall, Spearman correlation of mask area with the predicted spoiled probability, median
mask area of correct and misclassified images) to <out>.json.

Usage:
    python -m src.viz.mask_area_figure --predictions runs/results/predictions.csv \
        --segmentation data/processed/full/segmentation.csv --out figures/fig4_mask_area.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})

INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
PRETRAINED_COLOUR = "#A6A6A6"
# MeatGuard-S variants: Okabe-Ito blue, vermilion, bluish green
STUDENTS = (("meatguard_s:scratch", "MeatGuard-S, no distillation", "#0072B2", "o"),
            ("meatguard_s:distilled", "MeatGuard-S, distilled", "#D55E00", "s"),
            ("meatguard_s:pruned", "MeatGuard-S, pruned", "#009E73", "^"))
CLASS_COLOURS = {0: ("Fresh", "#0072B2"), 1: ("Spoiled", "#D55E00")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="runs/results/predictions.csv")
    ap.add_argument("--segmentation", default="data/processed/full/segmentation.csv")
    ap.add_argument("--out", default="figures/fig4_mask_area.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions)
    area = pd.read_csv(args.segmentation).set_index("path")["mask_area"]
    pred["mask_area"] = pred["path"].map(area)
    assert pred["mask_area"].notna().all(), "predictions contain images missing from the segmentation file"
    pred["pred"] = (pred["p_spoiled"] >= 0.5).astype(int)
    models = list(dict.fromkeys(pred["model"]))
    pretrained = [m for m in models if m.startswith(("teacher:", "baseline:"))]

    ref = pred[pred["model"] == models[0]]
    spoiled = ref[ref["label"] == 1]
    edges = np.quantile(spoiled["mask_area"], [0, 0.25, 0.5, 0.75, 1])
    quartile = lambda a: np.clip(np.searchsorted(edges, a, side="right") - 1, 0, 3)

    stats = {"auc_one_minus_mask_area": float(roc_auc_score(ref["label"], 1 - ref["mask_area"])),
             "quartile_edges": edges.round(3).tolist(), "models": {}}
    for m in models:
        d = pred[(pred["model"] == m) & (pred["label"] == 1)].copy()
        d["q"] = quartile(d["mask_area"])
        recall = (d["pred"] == 1).groupby(d["q"]).mean()
        correct = d["pred"] == 1
        stats["models"][m] = {
            "recall_by_quartile": recall.round(3).tolist(),
            "spearman_area_vs_p_spoiled": float(spearmanr(d["mask_area"], d["p_spoiled"])[0]),
            "median_area_spoiled_correct": float(d.loc[correct, "mask_area"].median()),
            "median_area_spoiled_missed": float(d.loc[~correct, "mask_area"].median()) if (~correct).any() else None}

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.48, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(4)
    for m in pretrained:
        a.plot(x, stats["models"][m]["recall_by_quartile"], color=PRETRAINED_COLOUR, lw=1.2, zorder=1)
    a.plot([], [], color=PRETRAINED_COLOUR, lw=1.2, label=f"ImageNet-pretrained (teacher, {len(pretrained) - 1} baselines)")
    for m, label, colour, marker in STUDENTS:
        if m in stats["models"]:
            a.plot(x, stats["models"][m]["recall_by_quartile"], color=colour, lw=1.6, marker=marker, ms=4.5,
                   mec="white", mew=0.6, label=label, zorder=3)
    a.set_xticks(x, [f"Q{i + 1}\n{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(4)])
    a.set_xlabel("Mask area (fraction of frame), quartiles of spoiled test images", color=INK)
    a.set_ylabel("Spoiled-class recall", color=INK)
    a.legend(frameon=False, loc="lower left")

    for label, (name, colour) in CLASS_COLOURS.items():
        b.hist(ref.loc[ref["label"] == label, "mask_area"], bins=np.linspace(0, 1, 21), density=True,
               histtype="step", lw=1.6, color=colour, label=name)
    b.set_xlabel("Mask area (fraction of frame)", color=INK)
    b.set_ylabel("Density (test images)", color=INK)
    b.legend(frameon=False, loc="upper left")

    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, length=2.5, width=0.6)
        ax.set_title(tag, loc="left", fontsize=8, color=INK)

    fig.tight_layout(w_pad=2.0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
