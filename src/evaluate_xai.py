"""
Quantitative evaluation of the explanations (Section 3.6 of the paper).

1. Grad-CAM layer selection on a *validation* sample: the candidate layer with the largest mean
   (insertion AUC - deletion AUC) is used for the test evaluation, so the test set does not choose the layer.
2. On a session-stratified *test* sample, for Grad-CAM, occlusion sensitivity and SHAP (SLIC superpixels):
   deletion and insertion AUC (Eq. 7) and the same areas for a random pixel order; the meat share (Eq. 8)
   against the mask area, using the SAM masks in <data>/../masks/ transformed by the image crop.
3. Cascading weight randomisation (Adebayo et al., 2018) for Grad-CAM and occlusion on a subset.

Writes to --out (default runs/xai/): layer_selection.json, xai_per_image.csv, sanity_check.csv,
xai_summary.json (all settings and per-method / per-class means) and example_maps.npz (maps for figures).

Usage:  python -m src.evaluate_xai --data data/processed/full/crops --model runs/student/pruned/best_model.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import find_split_file
from src.data.segment_meat import load_cropped_mask
from src.data.transforms import get_val_test_transform
from src.train_student import environment, load_student
from src.train_teacher import resolve_device, seed_everything, sha256
from src.xai.faithfulness import (curve_auc, meat_share, perturbation_curve, random_order_auc,
                                  randomization_sanity_check)
from src.xai.gradcam import GradCAM
from src.xai.occlusion import OcclusionSensitivity

LAYERS = ("stage2", "s3_attn", "stage4")


def stratified_sample(table, split, per_class, per_session, seed):
    """Up to `per_class` images per class from `split`, at most `per_session` per capture session, so that a
    few large sessions do not dominate the sample. With a specimen-level split file, whose `session` column
    holds specimen groups, the cap applies to the capture sessions (`capture_session`) within them."""
    rng = np.random.default_rng(seed)
    col = "capture_session" if "capture_session" in table.columns else "session"
    rows = []
    for cls, part in table[table["split"] == split].groupby("class"):
        picked = []
        sessions = part[col].unique()
        rng.shuffle(sessions)
        for s in sessions:
            frames = part[part[col] == s]
            picked.append(frames.sample(min(per_session, len(frames)), random_state=int(rng.integers(1 << 31))))
            if sum(map(len, picked)) >= per_class:
                break
        part = pd.concat(picked).head(per_class).reset_index(drop=True)
        rows.append(part.assign(_rank=np.arange(len(part)), _cls=len(rows)))
    # Interleave the classes (fresh, spoiled, fresh, ...), so that the first images (sanity check, example maps)
    # cover both. An explicit sort: the earlier MultiIndex swaplevel/sort_index did not interleave under every
    # pandas version.
    return (pd.concat(rows).sort_values(["_rank", "_cls"], kind="stable")
            .drop(columns=["_rank", "_cls"]).reset_index(drop=True))


class Explainers:
    def __init__(self, model, device, background=None, shap_samples=100, patch=16, stride=8):
        self.model, self.device = model, device
        self.occlusion = OcclusionSensitivity(model, patch_size=patch, stride=stride)
        self.shap = None
        if background is not None:
            from src.xai.shap_module import MeatGuardSHAP
            self.shap = MeatGuardSHAP(model, background)
            self.shap_samples = shap_samples

    def gradcam(self, model, x, cls, layer):
        cam = GradCAM(model, target_layer=layer)
        try:
            return cam.generate(x, class_idx=cls)
        finally:
            cam.remove_hooks()

    def occ(self, model, x, cls):
        return OcclusionSensitivity(model, self.occlusion.patch_size, self.occlusion.stride).generate_map(
            x, class_idx=cls, normalize=False)

    def shap_map(self, x, cls):
        values = self.shap.explain(x, n_samples=self.shap_samples)
        return self.shap.superpixel_attribution(x, values, labels=cls)[2]


def faithfulness(model, x, attribution, cls):
    t, f_del = perturbation_curve(model, x, attribution, cls, "deletion")
    _, f_ins = perturbation_curve(model, x, attribution, cls, "insertion")
    return {"deletion_auc": curve_auc(t, f_del), "insertion_auc": curve_auc(t, f_ins)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="crops/ ImageFolder root")
    ap.add_argument("--masks", default=None, help="masks/ root (default: <data>/../masks)")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--model", default="runs/student/pruned/best_model.pt")
    ap.add_argument("--out", default="runs/xai")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-per-class", type=int, default=50, help="validation images per class for layer selection")
    ap.add_argument("--test-per-class", type=int, default=100, help="test images per class (0 = all test images)")
    ap.add_argument("--per-session", type=int, default=2, help="max images per capture session in the samples")
    ap.add_argument("--shap-background", type=int, default=50, help="training images in the SHAP background set")
    ap.add_argument("--shap-samples", type=int, default=100)
    ap.add_argument("--sanity-images", type=int, default=20)
    ap.add_argument("--random-repeats", type=int, default=3)
    ap.add_argument("--layer", default=None, help="skip layer selection and use this Grad-CAM layer")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data)
    masks_dir = Path(args.masks) if args.masks else data_dir.parent / "masks"
    split_file = Path(args.split_file) if args.split_file else find_split_file(data_dir)
    table = pd.read_csv(split_file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_student(args.model, device).to(device).eval()
    tf = get_val_test_transform()
    from PIL import Image
    load = lambda rel: tf(Image.open(data_dir / rel).convert("RGB")).unsqueeze(0).to(device)
    label_of = {c: i for i, c in enumerate(sorted(table["class"].unique()))}

    @torch.no_grad()
    def prediction(x):
        p = torch.softmax(model(x), dim=1)[0]
        return int(p.argmax()), float(p.max())

    # ---- 1. Grad-CAM layer selection on validation images
    ex = Explainers(model, device)
    if args.layer:
        layer, layer_scores = args.layer, None
    else:
        val = stratified_sample(table, "val", args.val_per_class, args.per_session, args.seed)
        scores = {l: [] for l in LAYERS}
        for rel in val["path"]:
            x = load(rel)
            cls, _ = prediction(x)
            for l in LAYERS:
                f = faithfulness(model, x, ex.gradcam(model, x, cls, l), cls)
                scores[l].append(f["insertion_auc"] - f["deletion_auc"])
        layer_scores = {l: float(np.mean(v)) for l, v in scores.items()}
        layer = max(layer_scores, key=layer_scores.get)
        (out / "layer_selection.json").write_text(json.dumps(
            {"candidates": layer_scores, "selected": layer, "criterion": "mean(insertion AUC - deletion AUC)",
             "n_val_images": len(val)}, indent=2))
        print(f"Grad-CAM layer scores (validation): {layer_scores} -> {layer}")

    # ---- 2. Test evaluation
    test = (table[table["split"] == "test"].reset_index(drop=True) if args.test_per_class == 0 else
            stratified_sample(table, "test", args.test_per_class, args.per_session, args.seed))
    print(f"test sample: {len(test)} images, {test['session'].nunique()} sessions; classes of the first 10: "
          f"{test['class'].head(10).tolist()}")
    background = None
    if not args.no_shap:
        bg = table[table["split"] == "train"].sample(args.shap_background, random_state=args.seed)["path"]
        background = torch.cat([load(rel) for rel in bg])
    ex = Explainers(model, device, background, args.shap_samples)
    methods = {"gradcam": lambda x, c: ex.gradcam(model, x, c, layer), "occlusion": lambda x, c: ex.occ(model, x, c)}
    if ex.shap is not None:
        methods["shap"] = ex.shap_map

    rows, examples, t0 = [], {}, time.time()
    for i, rel in enumerate(test["path"]):
        x = load(rel)
        cls, conf = prediction(x)
        mask = load_cropped_mask((masks_dir / rel).with_suffix(".png"))
        base = {"path": rel, "session": test.loc[i, "session"], "label": label_of[test.loc[i, "class"]],
                "pred": cls, "confidence": conf,
                "random_deletion_auc": random_order_auc(model, x, cls, "deletion", repeats=args.random_repeats),
                "random_insertion_auc": random_order_auc(model, x, cls, "insertion", repeats=args.random_repeats)}
        for name, explain in methods.items():
            # SHAP samples randomly: reseed per image and method so each map is reproducible on its own,
            # independent of the sample order (src.xai_examples recomputes the maps the same way)
            seed_everything(args.seed)
            a = np.asarray(explain(x, cls), dtype=np.float32)
            rows.append({**base, "method": name, **faithfulness(model, x, a, cls), **meat_share(a, mask)})
            if i < 16:
                examples[f"{i}_{name}"] = a.astype(np.float16)
        if i < 16:
            examples[f"{i}_mask"] = mask
            examples[f"{i}_path"] = np.array(rel)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(test)} images  {(time.time() - t0) / (i + 1):.1f}s/img")
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(out / "xai_per_image.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "xai_per_image.csv", index=False)
    np.savez_compressed(out / "example_maps.npz", **examples)

    # ---- 3. Randomisation sanity check
    sanity = []
    for rel in test["path"][:args.sanity_images]:
        x = load(rel)
        cls, _ = prediction(x)
        for name, fn in (("gradcam", lambda m, x, c: ex.gradcam(m, x, c, layer)), ("occlusion", ex.occ)):
            for r in randomization_sanity_check(model, fn, x, cls):
                sanity.append({"path": rel, "method": name, **r})
    pd.DataFrame(sanity).to_csv(out / "sanity_check.csv", index=False)

    df["correct"] = df["label"] == df["pred"]
    df["pred_class"] = df["pred"].map({v: k for k, v in label_of.items()})
    agg = lambda g: g[["deletion_auc", "insertion_auc", "random_deletion_auc", "random_insertion_auc",
                       "meat_share", "mask_area"]].mean().round(4).to_dict()
    summary = {
        "model": str(args.model), "gradcam_layer": layer, "layer_selection": layer_scores,
        "settings": {"test_sample": "all" if args.test_per_class == 0 else
                     f"{args.test_per_class} per class, <= {args.per_session} per session",
                     "n_test_images": int(test.shape[0]), "n_test_sessions": int(test["session"].nunique()),
                     "occlusion": {"patch": 16, "stride": 8, "fill": "mean colour"},
                     "shap": None if args.no_shap else {"background_train_images": args.shap_background,
                                                       "samples": args.shap_samples, "slic_segments": 50},
                     "perturbation_steps": 10, "random_repeats": args.random_repeats,
                     "sanity_images": args.sanity_images, "seed": args.seed},
        "by_method": {m: agg(g) for m, g in df.groupby("method")},
        "by_method_and_predicted_class": {f"{m}|{c}": agg(g) for (m, c), g in df.groupby(["method", "pred_class"])},
        "sanity_mean_spearman": pd.DataFrame(sanity).groupby(["method", "randomised_up_to"], sort=False)["spearman"]
                                .mean().round(4).reset_index().to_dict("records") if sanity else None,
        "test_sample_accuracy": float(df.drop_duplicates("path")["correct"].mean()),
        "split_sha256": sha256(split_file), **environment(device),
    }
    (out / "xai_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["by_method"], indent=2))


if __name__ == "__main__":
    main()
