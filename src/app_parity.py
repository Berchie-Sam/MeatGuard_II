"""
Parity between the Android app and Python for the deployed TFLite model (app Requirements 7, test plan T2/T3).

prepare: writes an evaluation folder for the device and the expected values from Python.
    <out>/eval/crops/<class>/<name>.png    test-split crops (the training-pipeline JPEGs decoded by PIL and stored
                                           losslessly, so the app sees exactly the pixels Python sees)
    <out>/eval/photos/<class>/<name>.<ext> original photos of the same images
    <out>/eval/masks/<class>/<name>.png    their meat masks (255 = meat)
    <out>/expected_a.csv                   mode (a): logits and p_spoiled of the TFLite model on each crop
    <out>/expected_b.csv                   mode (b): crop window, mask area, checksum of the 224 crop made in memory by
                                           crop_to_mask (no JPEG step) and the TFLite p_spoiled on that crop
    Copy <out>/eval into the app's internal files/ folder (tar via adb run-as; see the device-runs notes).

compare: joins a batch CSV written by the app (results/batch_*.csv) with the expected values and reports the
    differences in p_spoiled, label agreement and the accuracy of both; for mode (b) also the crop windows and,
    if the app saved its crops, whether they are identical to Python's.

The model is run with tf.lite.Interpreter (TensorFlow 2.17), input normalised in float32 exactly as the app does:
(pixel / 255 - mean) / std.

Usage:
    python -m src.app_parity prepare --data E:/Datasets/full --raw E:/Datasets/MeatScan \
        --model D:/MeatGuard_phase0/models/meatguard_s_int8_float_attn.tflite --out D:/MeatGuard_eval
    python -m src.app_parity compare --expected D:/MeatGuard_eval/expected_a.csv --phone <batch_a_...csv>
"""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 224
BG_FILL = (124, 116, 104)
PAD = 0.08
SMALL_MASK = 0.02
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = {"Fresh_CowMeat": 0, "Spoiled_CowMeat": 1}


# Same as src/data/segment_meat.py (copied so that this script does not need torch).
def crop_window(mask, pad=PAD):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = min(int(max(y1 - y0, x1 - x0) * (1 + 2 * pad)), max(h, w))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    x0 = int(np.clip(cx - side // 2, 0, max(w - side, 0)))
    y0 = int(np.clip(cy - side // 2, 0, max(h - side, 0)))
    return x0, y0, side


def crop_to_mask(masked, mask, pad=PAD):
    x0, y0, side = crop_window(mask, pad)
    canvas = Image.new("RGB", (side, side), BG_FILL)
    canvas.paste(Image.fromarray(masked), (-x0, -y0))
    return canvas.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


class Model:
    def __init__(self, path):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=str(path), num_threads=4)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        assert tuple(self.inp["shape"]) == (1, IMG_SIZE, IMG_SIZE, 3), self.inp["shape"]

    def __call__(self, rgb224):
        x = (np.asarray(rgb224, np.uint8).astype(np.float32) / np.float32(255) - MEAN) / STD
        self.interp.set_tensor(self.inp["index"], x[None].astype(np.float32))
        self.interp.invoke()
        z = self.interp.get_tensor(self.out["index"])[0].astype(np.float64)
        e = np.exp(z - z.max())
        return z, float(e[1] / e.sum())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rgb, np.uint8)).tobytes()).hexdigest()[:16]


def prepare(args):
    import pandas as pd
    data, raw, out = Path(args.data), Path(args.raw), Path(args.out)
    table = pd.read_csv(args.split_file or data / "specimens.csv")
    test = table[table["split"] == args.split].sort_values("path").reset_index(drop=True)
    if args.limit:
        test = test.groupby("class", group_keys=False).head(args.limit // 2)
    print(f"{len(test)} {args.split} images; model {args.model} (sha256 {sha256(args.model)[:12]}...)")
    model = Model(args.model)

    rows_a, rows_b = [], []
    for i, r in enumerate(test.itertuples()):
        rel = Path(r.path)
        stem = rel.with_suffix("")
        label = CLASSES[rel.parts[0]]
        # Mode (a): the stored crop, as used in evaluation
        crop = Image.open((data / "crops" / rel).with_suffix(".jpg")).convert("RGB")
        dst = out / "eval" / "crops" / stem.with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst)
        z, p = model(crop)
        rows_a.append({"file": stem.with_suffix(".png").as_posix(), "label": label, "logit_fresh": z[0],
                       "logit_spoiled": z[1], "p_spoiled": p})
        # Mode (b): original photo + mask -> fill + crop in memory -> model
        if not args.no_photos:
            photo_src = raw / rel
            mask_src = (data / "masks" / rel).with_suffix(".png")
            rgb = np.asarray(Image.open(photo_src).convert("RGB"))
            mask = np.asarray(Image.open(mask_src)) > 127
            assert mask.shape == rgb.shape[:2], (rel, mask.shape, rgb.shape)
            masked = rgb.copy()
            masked[~mask] = BG_FILL
            c224 = crop_to_mask(masked, mask)
            x0, y0, side = crop_window(mask)
            zb, pb = model(c224)
            pdst = out / "eval" / "photos" / rel
            mdst = out / "eval" / "masks" / stem.with_suffix(".png")
            pdst.parent.mkdir(parents=True, exist_ok=True)
            mdst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(photo_src, pdst)
            shutil.copyfile(mask_src, mdst)
            area = float(mask.mean())
            rows_b.append({"file": rel.as_posix(), "label": label, "width": rgb.shape[1], "height": rgb.shape[0],
                           "mask_area": area, "crop_left": x0, "crop_top": y0, "crop_side": side,
                           "no_meat": int(area < SMALL_MASK), "crop_sha": pixel_sha(c224),
                           "logit_fresh": zb[0], "logit_spoiled": zb[1], "p_spoiled": pb})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(test)}", flush=True)

    for name, rows in (("expected_a.csv", rows_a), ("expected_b.csv", rows_b)):
        if rows:
            with open(out / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    with open(out / "README.txt", "w", encoding="utf-8") as f:
        f.write(f"split={args.split} images={len(test)} model={Path(args.model).name} sha256={sha256(args.model)}\n"
                f"crops: training-pipeline JPEG crops decoded by Pillow {Image.__version__}, saved as PNG\n"
                "photos/masks: original photos and SAM masks; expected_b computed without EXIF rotation\n")
    acc = np.mean([(r["p_spoiled"] >= 0.5) == r["label"] for r in rows_a])
    print(f"Wrote {out}: expected_a.csv ({len(rows_a)} rows, accuracy {acc:.4f})"
          + (f", expected_b.csv ({len(rows_b)} rows)" if rows_b else ""))


def compare(args):
    import pandas as pd
    exp = pd.read_csv(args.expected)
    phone = pd.read_csv(args.phone, comment="#")
    meta = [l[2:].strip() for l in open(args.phone, encoding="utf-8") if l.startswith("# ")]
    m = exp.merge(phone, on="file", how="outer", suffixes=("_py", "_app"), indicator=True)
    missing = m[m["_merge"] != "both"]
    m = m[m["_merge"] == "both"]
    err = m[m["error"].notna() & (m["error"].astype(str) != "")] if "error" in m else m.iloc[:0]
    ok = m.drop(err.index)
    d = (ok["p_spoiled_app"] - ok["p_spoiled_py"]).abs()
    lab_py = ok["p_spoiled_py"] >= 0.5
    lab_app = ok["p_spoiled_app"] >= 0.5
    print("\n".join(l for l in meta if l.split(":")[0] in
                    ("manufacturer", "model", "soc_model", "android_version", "app_version", "model_sha256", "run")))
    print(f"rows: expected {len(exp)}, app {len(phone)}, matched {len(m)}, unmatched {len(missing)}, app errors {len(err)}")
    print(f"|p_app - p_py|: max {d.max():.2e}, mean {d.mean():.2e}, median {d.median():.2e}, "
          f"> 1e-3: {(d > 1e-3).sum()}, > 1e-2: {(d > 1e-2).sum()}")
    print(f"label agreement: {(lab_py == lab_app).mean():.4f} ({(lab_py != lab_app).sum()} differ)")
    # The app's CSV has its own "label" column (predicted FRESH/SPOILED), so the true label is label_py.
    truth = (ok["label_py"] if "label_py" in ok else ok["label"]).astype(bool)
    print(f"accuracy: python {(lab_py == truth).mean():.4f}, app {(lab_app == truth).mean():.4f}")
    if "crop_side_py" in ok:
        same_win = ((ok["crop_left_py"] == ok["crop_left_app"]) & (ok["crop_top_py"] == ok["crop_top_app"])
                    & (ok["crop_side_py"] == ok["crop_side_app"]))
        print(f"crop window identical: {same_win.mean():.4f} ({(~same_win).sum()} differ)")
        print(f"mask area max diff: {(ok['mask_area_py'] - ok['mask_area_app']).abs().max():.2e}")
        if args.crops:
            same = 0
            for r in ok.itertuples():
                f = Path(args.crops) / Path(r.file).with_suffix(".png")
                if f.exists() and pixel_sha(Image.open(f).convert("RGB")) == r.crop_sha:
                    same += 1
            print(f"app crops pixel-identical to Python: {same}/{len(ok)}")
    worst = ok.assign(diff=d).sort_values("diff", ascending=False).head(5)
    print("largest differences:")
    print(worst[["file", "p_spoiled_py", "p_spoiled_app", "diff"]].to_string(index=False))
    if args.out:
        ok.assign(diff=d).to_csv(args.out, index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True, help="segmentation output: crops/, masks/, specimens.csv")
    p.add_argument("--raw", required=True, help="original photos (same relative paths)")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split-file", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0, help="at most N images (half per class), for a quick check")
    p.add_argument("--no-photos", action="store_true", help="only mode (a)")
    c = sub.add_parser("compare")
    c.add_argument("--expected", required=True)
    c.add_argument("--phone", required=True)
    c.add_argument("--crops", default=None, help="folder of crops saved by the app (mode b)")
    c.add_argument("--out", default=None, help="write the joined table")
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else compare(args)


if __name__ == "__main__":
    main()
