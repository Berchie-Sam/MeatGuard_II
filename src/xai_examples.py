"""
Example explanations of fresh and spoiled test images for the paper's figure.

`evaluate_xai` stores maps only for the first images of its test sample, which are all fresh. This script
recomputes Grad-CAM, occlusion and SHAP, with the same settings, for a few images of *each* class taken from that
same sample (runs/xai/xai_per_image.csv), so the figure shows images whose numbers are in Table 7.

Selection (no hand-picking): per true class, correctly classified images from different capture sessions whose
Grad-CAM meat share minus mask area is closest to the class median, i.e. typical rather than best-looking cases.
Pass --paths to override.

Writes to --out (default runs/xai/examples/): maps.npz, selected.csv (per image and method: meat share and mask
area recomputed here and as stored by evaluate_xai) and the figure (--fig). With --redraw the figure is drawn again
from maps.npz and selected.csv without recomputing anything (no PyTorch needed, e.g. locally with the paper fonts).

Usage:
    python -m src.xai_examples --data data/processed/full/crops --model runs/student/pruned/best_model.pt
    python -m src.xai_examples --data data/processed/full/crops --redraw --fig figures/fig5_xai_examples.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

METHODS = (("gradcam", "Grad-CAM"), ("occlusion", "Occlusion"), ("shap", "SHAP"))
CLASS_NAME = {0: "Fresh", 1: "Spoiled"}


def select(per_image, n_per_class):
    g = per_image[per_image["method"] == "gradcam"].copy()
    g = g[g["label"] == g["pred"]]
    g["excess"] = g["meat_share"] - g["mask_area"]
    picked = []
    for _, part in g.groupby("label"):
        part = part.assign(dist=(part["excess"] - part["excess"].median()).abs()).sort_values("dist")
        picked.append(part.drop_duplicates("session").head(n_per_class))
    return pd.concat(picked)["path"].tolist()


def upsample(a, shape):
    """Bilinear resize of an attribution map for display (the meat shares are computed on the exact maps)."""
    a = np.asarray(a, dtype=np.float32)
    return a if a.shape == tuple(shape) else np.asarray(Image.fromarray(a).resize(shape[::-1], Image.BILINEAR))


def redraw(args, out):
    """The figure from maps.npz and selected.csv of an earlier run."""
    maps, sel = np.load(out / "maps.npz"), pd.read_csv(out / "selected.csv")
    records, i = [], 0
    while f"{i}_path" in maps.files:
        rel, mask = str(maps[f"{i}_path"]), maps[f"{i}_mask"]
        rows = sel[sel["path"] == rel].set_index("method")
        img = np.asarray(Image.open(Path(args.data) / rel).convert("RGB").resize(mask.shape[::-1]),
                         dtype=np.float32) / 255
        rec = {"path": rel, "label": int(rows["label"].iloc[0]), "pred": int(rows["pred"].iloc[0]),
               "confidence": float(rows["confidence"].iloc[0]), "image": img, "mask": mask,
               "mask_area": float(rows["mask_area"].iloc[0])}
        for key, _ in METHODS:
            rec[key] = upsample(maps[f"{i}_{key}"], mask.shape)
            rec[key + "_share"] = float(rows.loc[key, "meat_share"])
        records.append(rec)
        i += 1
    figure(records, Path(args.fig) if args.fig else out / "xai_examples_fresh_spoiled.png", args.dpi)


def figure(records, path, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    # Arial as in the other figures; Liberation Sans (metric-compatible, on Colab) or DejaVu Sans otherwise
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7, "axes.linewidth": 0.6})
    ink, mask_colour = "#222222", "#0072B2"
    # One hue (Okabe-Ito vermilion) whose opacity encodes attribution, over a grey copy of the image
    hue = np.array([0xD5, 0x5E, 0x00]) / 255
    legend_cmap = LinearSegmentedColormap.from_list("attr", ["#E6E6E6", "#D55E00"])

    rows, cols = len(records), 1 + len(METHODS)
    row_h, head = 1.16, 0.22
    fig, axes = plt.subplots(rows, cols, figsize=(5.5, row_h * rows + head), squeeze=False)
    for r, rec in enumerate(records):
        img = rec["image"]
        # Outline drawn from a smoothed copy of the mask (display only; the meat shares use the exact mask)
        mask = np.asarray(Image.fromarray(rec["mask"].astype(np.uint8) * 255).filter(ImageFilter.ModeFilter(9))) > 127
        grey = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2) * 0.55 + 0.45
        ax = axes[r, 0]
        ax.imshow(img)
        ax.contour(mask, levels=[0.5], colors=mask_colour, linewidths=0.8)
        ax.set_ylabel(f"{CLASS_NAME[rec['label']]}\nP = {rec['confidence']:.2f}", color=ink, fontsize=7)
        for c, (key, name) in enumerate(METHODS, start=1):
            ax = axes[r, c]
            a = np.clip(rec[key], 0, None)
            a = a / a.max() if a.max() > 0 else a
            overlay = grey * (1 - a[..., None]) + hue * a[..., None]
            ax.imshow(overlay)
            ax.contour(mask, levels=[0.5], colors=mask_colour, linewidths=0.8)
            ax.text(0.03, 0.04, f"on meat {100 * rec[key + '_share']:.0f}%", transform=ax.transAxes, fontsize=6,
                    color=ink, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].text(0.97, 0.04, f"meat {100 * rec['mask_area']:.0f}%", transform=axes[r, 0].transAxes,
                        ha="right", fontsize=6, color=ink,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    for c, name in enumerate(["Image and meat mask"] + [n for _, n in METHODS]):
        axes[0, c].set_title(name, fontsize=7, color=ink, pad=3)
    fig.subplots_adjust(left=0.07, right=0.90, top=1 - head / (row_h * rows + head), bottom=0.005,
                        wspace=0.04, hspace=0.04)
    cax = fig.add_axes([0.915, 0.25, 0.015, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=legend_cmap), cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["0", "max"], fontsize=6)
    cb.set_label("Positive attribution (per map)", fontsize=6)
    cb.outline.set_visible(False)
    fig.savefig(path, dpi=dpi)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="crops/ ImageFolder root")
    ap.add_argument("--masks", default=None, help="masks/ root (default: <data>/../masks)")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--model", default="runs/student/pruned/best_model.pt")
    ap.add_argument("--xai", default="runs/xai", help="folder written by evaluate_xai")
    ap.add_argument("--out", default=None, help="default: <xai>/examples")
    ap.add_argument("--fig", default=None, help="default: <out>/xai_examples_fresh_spoiled.png")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--paths", nargs="*", default=None, help="test images (relative to --data) instead of the rule")
    ap.add_argument("--shap-background", type=int, default=50)
    ap.add_argument("--shap-samples", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--redraw", action="store_true", help="only redraw the figure from <out>/maps.npz")
    args = ap.parse_args()
    if args.redraw:
        return redraw(args, Path(args.out) if args.out else Path(args.xai) / "examples")

    import torch
    from src.data.dataset import find_split_file
    from src.data.segment_meat import load_cropped_mask
    from src.data.transforms import get_val_test_transform
    from src.evaluate_xai import Explainers
    from src.train_student import load_student
    from src.train_teacher import resolve_device, seed_everything
    from src.xai.faithfulness import _to_hw, meat_share

    seed_everything(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data)
    masks_dir = Path(args.masks) if args.masks else data_dir.parent / "masks"
    split_file = Path(args.split_file) if args.split_file else find_split_file(data_dir)
    table = pd.read_csv(split_file)
    xai = Path(args.xai)
    out = Path(args.out) if args.out else xai / "examples"
    out.mkdir(parents=True, exist_ok=True)
    per_image = pd.read_csv(xai / "xai_per_image.csv")
    layer = json.loads((xai / "layer_selection.json").read_text())["selected"]
    paths = args.paths or select(per_image, args.per_class)
    print(f"Grad-CAM layer: {layer}; images: {paths}")

    model = load_student(args.model, device).to(device).eval()
    tf = get_val_test_transform()
    load = lambda rel: tf(Image.open(data_dir / rel).convert("RGB")).unsqueeze(0).to(device)
    # Same SHAP background and settings as evaluate_xai
    bg = table[table["split"] == "train"].sample(args.shap_background, random_state=args.seed)["path"]
    ex = Explainers(model, device, torch.cat([load(rel) for rel in bg]), args.shap_samples)
    explain = {"gradcam": lambda x, c: ex.gradcam(model, x, c, layer),
               "occlusion": lambda x, c: ex.occ(model, x, c), "shap": ex.shap_map}

    records, rows, maps = [], [], {}
    for i, rel in enumerate(paths):
        x = load(rel)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[0]
        cls = int(p.argmax())
        mask = load_cropped_mask((masks_dir / rel).with_suffix(".png"))
        label = int(rel.split("/")[0].lower().startswith("spoiled"))
        img = np.asarray(Image.open(data_dir / rel).convert("RGB").resize(mask.shape[::-1]), dtype=np.float32) / 255
        rec = {"path": rel, "label": label, "pred": cls, "confidence": float(p[cls]), "image": img, "mask": mask}
        for key, _ in METHODS:
            seed_everything(args.seed)
            a = np.asarray(explain[key](x, cls), dtype=np.float32)
            rec[key] = _to_hw(a, *mask.shape)
            m = meat_share(a, mask)
            rec[key + "_share"], rec["mask_area"] = m["meat_share"], m["mask_area"]
            stored = per_image[(per_image["path"] == rel) & (per_image["method"] == key)]
            rows.append({"path": rel, "label": label, "pred": cls, "confidence": rec["confidence"], "method": key,
                         "meat_share": m["meat_share"], "mask_area": m["mask_area"],
                         "meat_share_evaluate_xai": float(stored["meat_share"].iloc[0]) if len(stored) else None})
            maps[f"{i}_{key}"] = a.astype(np.float16)
        maps[f"{i}_mask"], maps[f"{i}_path"] = mask, np.array(rel)
        records.append(rec)
        print(f"{rel}: {CLASS_NAME[label]} -> {CLASS_NAME[cls]} ({rec['confidence']:.2f}); "
              + ", ".join(f"{k} {rec[k + '_share']:.2f}" for k, _ in METHODS) + f"; mask {rec['mask_area']:.2f}")

    sel = pd.DataFrame(rows)
    sel.to_csv(out / "selected.csv", index=False)
    np.savez_compressed(out / "maps.npz", **maps)
    print(sel[["path", "method", "meat_share", "meat_share_evaluate_xai"]].to_string(index=False))
    figure(records, Path(args.fig) if args.fig else out / "xai_examples_fresh_spoiled.png", args.dpi)


if __name__ == "__main__":
    main()
