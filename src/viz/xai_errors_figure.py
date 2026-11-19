"""
Explanations of misclassified test images (Section 4.3 of the paper).

Uses the maps that src.evaluate_xai stores for the first images of its test sample (example_maps.npz), so nothing is
recomputed: the misclassified images among them, at most one per specimen group, drawn as in Figure 5 by
src.xai_examples, with the true and the predicted class on each row.

Usage:
    python -m src.viz.xai_errors_figure --data data/processed/full/crops --xai runs/xai \
        --out figures/fig_xai_errors.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.xai_examples import CLASS_NAME, METHODS, figure, upsample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="crops/ ImageFolder root")
    ap.add_argument("--xai", default="runs/xai")
    ap.add_argument("--out", default="figures/fig_xai_errors.png")
    ap.add_argument("--max-rows", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    xai = Path(args.xai)
    maps = np.load(xai / "example_maps.npz")
    per_image = pd.read_csv(xai / "xai_per_image.csv")

    records, groups, i = [], set(), 0
    while f"{i}_path" in maps.files and len(records) < args.max_rows:
        rel, mask = str(maps[f"{i}_path"]), maps[f"{i}_mask"]
        rows = per_image[per_image["path"] == rel].set_index("method")
        first = rows.iloc[0]
        i += 1
        if first["label"] == first["pred"] or first["session"] in groups:
            continue
        groups.add(first["session"])
        img = np.asarray(Image.open(Path(args.data) / rel).convert("RGB").resize(mask.shape[::-1]),
                         dtype=np.float32) / 255
        rec = {"path": rel, "label": int(first["label"]), "pred": int(first["pred"]),
               "confidence": float(first["confidence"]), "image": img, "mask": mask,
               "mask_area": float(first["mask_area"]),
               "row_label": f"{CLASS_NAME[int(first['label'])]} → {CLASS_NAME[int(first['pred'])].lower()}\n"
                            f"P = {float(first['confidence']):.2f}"}
        for key, _ in METHODS:
            rec[key] = upsample(maps[f"{i - 1}_{key}"].astype(np.float32), mask.shape)
            rec[key + "_share"] = float(rows.loc[key, "meat_share"])
        records.append(rec)
        print(f"{rel}: group {first['session']}, true {first['label']}, predicted {first['pred']}, "
              f"P = {first['confidence']:.2f}, mask area {first['mask_area']:.2f}")
    figure(records, Path(args.out), args.dpi)


if __name__ == "__main__":
    main()
