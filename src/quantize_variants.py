"""
Diagnose the accuracy lost by full-integer (INT8) quantisation of MeatGuard-S and build alternatives.

    1. Per-layer quantisation error (TFLite QuantizationDebugger) on calibration images, written to
       layer_errors.csv; the layers with the largest error relative to their quantisation step are printed.
    2. Variants, each converted from the SavedModel written by `train_student export`:
         int8_full        the existing <name>_int8.tflite (int8 weights, activations, input and output)
         int8_float_attn  int8 except the attention block (s3_attn), which stays float; float input/output
         int8_float_worst int8 except the layers flagged by the debugger (RMSE / scale > threshold)
         int16x8          int8 weights with int16 activations; float input/output
    3. Each variant is compared with the pruned FP32 PyTorch model on the validation split: macro-F1,
       spoiled recall, prediction agreement and file size (summary.json, summary.csv).

Model choice uses the validation split only; run `python -m src.evaluate` afterwards for test results.

Usage (from the repository root, after `train_student export`):
    python -m src.quantize_variants --data data/processed/full/crops
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import make_loaders
from src.train_student import environment, load_student, resolve_split
from src.train_teacher import seed_everything, sha256
from src.utils.export import _calibration_images, _channels_last, tflite_predict
from src.utils.metrics import classification_metrics, predict


def representative(images):
    def gen():
        for img in images:
            yield [img[None, ...]]
    return gen


def layer_errors(converter, debug_data, path):
    """Per-layer statistics of the quantised model against its float version; rows sorted by RMSE / scale."""
    import tensorflow as tf
    debugger = tf.lite.experimental.QuantizationDebugger(converter=converter, debug_dataset=debug_data)
    debugger.run()
    with open(path, "w") as f:
        debugger.layer_statistics_dump(f)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        scale = float(r["scale"]) if r.get("scale") else 0.0
        mse = float(r["mean_squared_error"]) if r.get("mean_squared_error") else float("nan")
        ratio = np.sqrt(mse) / scale if scale > 0 else float("nan")
        # No statistic (e.g. a LayerNorm Rsqrt whose output range collapses) is treated as the worst case
        r["rmse_over_scale"] = float("inf") if np.isnan(ratio) else float(ratio)
    rows.sort(key=lambda r: r["rmse_over_scale"], reverse=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def selective(converter, debug_data, nodes):
    """INT8 model with the given nodes left in float."""
    import tensorflow as tf
    options = tf.lite.experimental.QuantizationDebugOptions(denylisted_nodes=nodes)
    debugger = tf.lite.experimental.QuantizationDebugger(converter=converter, debug_dataset=debug_data,
                                                         debug_options=options)
    return debugger.get_nondebug_quantized_model()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="ImageFolder root of the cropped images")
    ap.add_argument("--split-file", default=None, help="sessions.csv (default: next to --data)")
    ap.add_argument("--model", default="runs/student/pruned/best_model.pt")
    ap.add_argument("--export", default="runs/student/export", help="folder written by train_student export")
    ap.add_argument("--out", default=None, help="default: <export>/quant_variants")
    ap.add_argument("--name", default="meatguard_s")
    ap.add_argument("--calibration-images", type=int, default=500)
    ap.add_argument("--debug-images", type=int, default=100, help="calibration images used by the debugger")
    ap.add_argument("--threshold", type=float, default=0.7, help="RMSE / scale above which a layer stays float")
    ap.add_argument("--attn-pattern", default="s3_attn,layer_normalization,linalg.matmul,softmax",
                    help="comma-separated tensor-name substrings of the attention block (onnx2tf renames layers)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    import tensorflow as tf
    seed_everything(args.seed)
    data_dir, split_file, _ = resolve_split(args)
    export_dir = Path(args.export)
    saved_model = export_dir / "saved_model"
    out = Path(args.out) if args.out else export_dir / "quant_variants"
    out.mkdir(parents=True, exist_ok=True)

    # Same loaders as `train_student export`: calibration on training images without augmentation
    calib_loader, val_loader, _, _ = make_loaders(data_dir, strategy="baseline", batch_size=32, seed=args.seed,
                                                  split_file=split_file, num_workers=args.num_workers)
    float_tflite = sorted(saved_model.glob("*float32.tflite"))
    nhwc = _channels_last(tuple(tf.lite.Interpreter(model_path=str(float_tflite[0])).get_input_details()[0]["shape"])) \
        if float_tflite else True
    # One fixed set of calibration images for every variant (the training loader reshuffles on each pass)
    calib = [img.transpose(1, 2, 0) if nhwc else img
             for img in _calibration_images(calib_loader, args.calibration_images)]
    print(f"[calibration] {len(calib)} training images, layout {'NHWC' if nhwc else 'NCHW'}")

    def converter(int16=False):
        c = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = representative(calib)
        if int16:
            c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                                           tf.lite.OpsSet.TFLITE_BUILTINS]
        return c

    debug_data = representative(calib[:args.debug_images])

    # 1. Per-layer error
    print("[debug] per-layer quantisation error ...")
    rows = layer_errors(converter(), debug_data, out / "layer_errors.csv")
    print(f"{'RMSE/scale':>10}  {'op':<22} tensor")
    for r in rows[:15]:
        print(f"{r['rmse_over_scale']:>10.3f}  {r['op_name']:<22} {r['tensor_name']}")
    patterns = [p.strip().lower() for p in args.attn_pattern.split(",") if p.strip()]
    attn_nodes = [r["tensor_name"] for r in rows if any(p in r["tensor_name"].lower() for p in patterns)]
    worst_nodes = [r["tensor_name"] for r in rows if r["rmse_over_scale"] > args.threshold]
    print(f"[debug] {len(attn_nodes)} tensors match {patterns}; "
          f"{len(worst_nodes)} have RMSE/scale > {args.threshold}")
    for key, nodes in (("int8_float_attn", attn_nodes), ("int8_float_worst", worst_nodes)):
        print(f"[debug] kept in float for {key}:")
        for n in nodes:
            print(f"          {n}")
    if not attn_nodes:
        print("[debug] no tensor name matches the attention pattern; check layer_errors.csv and pass --attn-pattern")

    # 2. Variants
    variants = {"int8_full": export_dir / f"{args.name}_int8.tflite"}
    builds = {"int8_float_attn": (lambda: selective(converter(), debug_data, attn_nodes)) if attn_nodes else None,
              "int8_float_worst": (lambda: selective(converter(), debug_data, worst_nodes)) if worst_nodes else None,
              "int16x8": lambda: converter(int16=True).convert()}
    for key, build in builds.items():
        if build is None:
            print(f"[{key}] skipped: no layers to keep in float")
            continue
        print(f"[{key}] converting ...")
        try:
            path = out / f"{args.name}_{key}.tflite"
            path.write_bytes(build())
            variants[key] = path
        except Exception as e:                          # one failed variant should not stop the others
            print(f"[{key}] failed: {type(e).__name__}: {e}")

    # 3. Validation comparison with the FP32 model
    model = load_student(args.model, "cpu")
    y, p_fp32 = predict(model, val_loader, torch.device("cpu"))
    size = lambda f: round(os.path.getsize(f) / 2 ** 20, 3)
    results = {"fp32_pytorch": {"size_mb": size(args.model), **classification_metrics(y, p_fp32),
                                "agreement_with_fp32": 1.0}}
    for key, path in variants.items():
        if not Path(path).exists():
            print(f"[{key}] missing: {path}")
            continue
        print(f"[{key}] validating ...")
        try:
            y_t, p = tflite_predict(path, val_loader)
        except Exception as e:                          # e.g. an op without a kernel for this data type
            print(f"[{key}] cannot run in TFLite: {type(e).__name__}: {e}")
            results[key] = {"size_mb": size(path), "error": f"{type(e).__name__}: {e}"}
            continue
        assert (y_t == y).all()
        results[key] = {"size_mb": size(path), **classification_metrics(y_t, p),
                        "agreement_with_fp32": float((p.argmax(1) == p_fp32.argmax(1)).mean())}

    cols = ["size_mb", "accuracy", "macro_f1", "recall_positive", "roc_auc", "agreement_with_fp32"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + cols + ["error"])
        for key, r in results.items():
            w.writerow([key] + [r.get(c) for c in cols] + [r.get("error", "")])
    summary = {"model": str(args.model), "saved_model": str(saved_model), "split_sha256": sha256(split_file),
               "calibration_images": args.calibration_images, "debug_images": args.debug_images,
               "threshold": args.threshold, "float_nodes": {"int8_float_attn": attn_nodes,
                                                            "int8_float_worst": worst_nodes},
               "validation": results, **environment(torch.device("cpu"))}
    summary["versions"]["tensorflow"] = tf.__version__
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'variant':<18}" + "".join(f"{c:>21}" for c in cols))
    for key, r in results.items():
        print(f"{key:<18}" + "".join(f"{r.get(c, float('nan')):>21.4f}" for c in cols)
              + (f"  ({r['error'][:60]})" if "error" in r else ""))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
