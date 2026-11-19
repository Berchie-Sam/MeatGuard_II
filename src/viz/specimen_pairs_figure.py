"""
Same specimens across capture sessions (Section 3.2.3 of the paper).

Pairs of a test image and a training image of the same class that src.data.specimen_check matched geometrically
under the split by capture session (session_pairs.csv: the best match of each test session). Selection (no
hand-picking): per class, the pairs at the 85th, 50th and 15th percentile of the inlier count among pairs with at
least --min-inliers inliers; for the median pair of each class a match verified in the mirrored orientation
(inliers_flipped) is used when one exists, so the figure shows re-photographed and mirrored copies alike.

Usage:
    python -m src.viz.specimen_pairs_figure --pairs runs/specimen_check/session_pairs.csv \
        --data data/processed/full/crops --out figures/fig_specimen_pairs.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})
INK, MUTED = "#222222", "#666666"
CLASSES = (("Fresh_CowMeat", "Fresh"), ("Spoiled_CowMeat", "Spoiled"))
QUANTILES = (0.85, 0.50, 0.15)


def select(pairs, min_inliers):
    picked = []
    for cls, _ in CLASSES:
        part = pairs[(pairs["query_class"] == cls) & (pairs["inliers"] >= min_inliers)].sort_values("inliers")
        for q in QUANTILES:
            target = part["inliers"].quantile(q)
            pool = part[part["inliers_flipped"]] if q == 0.50 and part["inliers_flipped"].any() else part
            row = pool.iloc[(pool["inliers"] - target).abs().argsort()[:1]]
            part = part.drop(row.index)
            picked.append(row)
    return pd.concat(picked)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="runs/specimen_check/session_pairs.csv")
    ap.add_argument("--data", required=True, help="crops/ ImageFolder root")
    ap.add_argument("--out", default="figures/fig_specimen_pairs.png")
    ap.add_argument("--min-inliers", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    pairs = pd.read_csv(args.pairs)
    pairs = pairs[(pairs["pair"] == "test->train") & (pairs["pool"] == "same_class")]
    chosen = select(pairs, args.min_inliers)

    n_rows, n_pairs = len(QUANTILES), len(CLASSES)
    fig, axes = plt.subplots(n_rows, 2 * n_pairs, figsize=(6.3, 1.72 * n_rows + 0.25), squeeze=False)
    for k, (_, row) in enumerate(chosen.iterrows()):
        r, c = k % n_rows, 2 * (k // n_rows)
        for j, (rel, part, session) in enumerate(((row["query"], "Test", row["query_session"]),
                                                  (row["match"], "Training", row["match_session"]))):
            ax = axes[r, c + j]
            ax.imshow(Image.open(Path(args.data) / rel).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(f"{part}, session {session}", fontsize=6, color=MUTED, labelpad=1.5)
        note = f"{row['inliers']} inliers" + (", mirrored" if row["inliers_flipped"] else "")
        axes[r, c].annotate(note, xy=(1.02, 1.04), xycoords="axes fraction", ha="center", va="bottom",
                            fontsize=6.5, color=INK)
        print(f"{row['query']} (s{row['query_session']}) <-> {row['match']} (s{row['match_session']}): "
              f"{row['inliers']} inliers, mirrored={row['inliers_flipped']}, cosine={row['cosine']:.3f}")
    for p, (_, name) in enumerate(CLASSES):
        axes[0, 2 * p].annotate(name, xy=(1.02, 1.22), xycoords="axes fraction", ha="center", va="bottom",
                                fontsize=7.5, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.04, wspace=0.05, hspace=0.42)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
