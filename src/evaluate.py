"""
Test-set evaluation of every trained model, for the results tables of the paper.

Collects the models found under --runs:
    teacher      runs/teacher/teacher_selection.json  (the selected teacher)
    baselines    runs/baselines/<arch>/best.pt         (lightweight comparison models)
    scratch      runs/student/scratch/best.pt          (MeatGuard-S without distillation)
    distilled    runs/student/distilled/best.pt        (MeatGuard-S)
    pruned       runs/student/pruned/best_model.pt     (pruned + fine-tuned)
    int8         runs/student/export/*_int8.tflite     (deployed model)
and writes to runs/results/:
    test_results.json   per model: accuracy, macro-F1, spoiled recall, ROC-AUC, confusion matrix, 95% CIs from
                        resampling test sessions, parameters, MACs, file size, CPU latency; McNemar tests
    test_results.csv    the same as a flat table
    predictions.csv     one row per (model, test image): path, session, label, P(spoiled)

Usage:  python -m src.evaluate --data E:/Datasets/full/crops --runs runs
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import find_split_file, make_loaders
from src.model.teacher import load_teacher
from src.train_student import load_student, model_cost
from src.train_teacher import resolve_device, session_groups, sha256
from src.utils.metrics import bootstrap_ci, classification_metrics, mcnemar_test, predict


@torch.no_grad()
def cpu_latency_ms(model, runs=100, warmup=10, threads=1):
    """Median single-image latency on this machine's CPU (a reference only; on-device latency is measured
    in the Android app)."""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.eval().cpu()
    x = torch.randn(1, 3, 224, 224)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(old)
    return float(np.median(times))


def find_models(runs):
    runs, models = Path(runs), {}
    sel = runs / "teacher" / "teacher_selection.json"
    if sel.exists():
        s = json.loads(sel.read_text())
        ckpt = Path(s["checkpoint"]) if Path(s["checkpoint"]).exists() else runs / "teacher" / s["selected_teacher"] / "best.pt"
        models[f"teacher:{s['selected_teacher']}"] = ("torch-teacher", ckpt)
    for ckpt in sorted((runs / "baselines").glob("*/best.pt")):
        models[f"baseline:{ckpt.parent.name}"] = ("torch-teacher", ckpt)
    for name, rel in (("meatguard_s:scratch", "student/scratch/best.pt"),
                      ("meatguard_s:distilled", "student/distilled/best.pt"),
                      ("meatguard_s:pruned", "student/pruned/best_model.pt")):
        if (runs / rel).exists():
            models[name] = ("torch-student", runs / rel)
    for tfl in sorted((runs / "student" / "export").glob("*_int8.tflite")):
        models["meatguard_s:int8"] = ("tflite", tfl)
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None, help="default: <runs>/results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    data_dir = Path(args.data)
    split_file = Path(args.split_file) if args.split_file else find_split_file(data_dir)
    out = Path(args.out) if args.out else Path(args.runs) / "results"
    out.mkdir(parents=True, exist_ok=True)
    _, _, test_loader, _ = make_loaders(data_dir, strategy="baseline", batch_size=64, split_file=split_file,
                                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    groups = session_groups(test_loader.dataset, data_dir, split_file)
    paths = [Path(p).relative_to(data_dir).as_posix() for p, _ in test_loader.dataset.samples]

    models = find_models(args.runs)
    if not models:
        raise FileNotFoundError(f"No trained models under {args.runs}")
    results, preds, y_ref = {}, {}, None
    for name, (kind, path) in models.items():
        print(f"== {name}: {path}")
        if kind == "tflite":
            from src.utils.export import tflite_predict
            y, p = tflite_predict(path, test_loader)
            info = {"params_m": None, "gmacs": None}
        else:
            model = load_teacher(path, map_location=device)[0] if kind == "torch-teacher" else load_student(path, device)
            model.to(device)
            y, p = predict(model, test_loader, device)
            info = model_cost(model.cpu())
            if not args.no_latency:
                info["cpu_latency_ms"] = cpu_latency_ms(model)
        y_ref = y if y_ref is None else y_ref
        assert (y == y_ref).all(), "test order differs between models"
        m = classification_metrics(y, p)
        for metric in ("accuracy", "macro_f1", "recall_positive"):
            m[f"{metric}_ci95_sessions"] = bootstrap_ci(y, p, groups, metric)
        m.update(info, file=str(path), file_size_mb=round(os.path.getsize(path) / 2 ** 20, 3))
        results[name], preds[name] = m, p
        print(f"   acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f} "
              f"[{m['macro_f1_ci95_sessions'][0]:.3f}, {m['macro_f1_ci95_sessions'][1]:.3f}]  "
              f"spoiled recall {m['recall_positive']:.4f}  AUC {m['roc_auc']:.4f}")

    # Paired comparisons on the same test images
    tests = []
    ref = "meatguard_s:distilled"
    pairs = [(ref, other) for other in preds if other != ref] if ref in preds else []
    if "meatguard_s:pruned" in preds and "meatguard_s:int8" in preds:
        pairs.append(("meatguard_s:pruned", "meatguard_s:int8"))
    for a, b in pairs:
        tests.append({"a": a, "b": b, **mcnemar_test(y_ref, preds[a].argmax(1), preds[b].argmax(1))})

    summary = {"models": results, "mcnemar": tests, "n_test_images": int(len(y_ref)),
               "n_test_sessions": int(len(set(groups))), "split_file": str(split_file),
               "split_sha256": sha256(split_file), "latency_device": platform.processor() or "cpu",
               "latency_threads": 1}
    (out / "test_results.json").write_text(json.dumps(summary, indent=2))
    flat = []
    for name, m in results.items():
        row = {"model": name}
        for k, v in m.items():
            if isinstance(v, (list, tuple)) and k.endswith("_ci95_sessions"):
                row[k + "_lo"], row[k + "_hi"] = v
            elif k != "confusion":
                row[k] = v
        flat.append(row)
    pd.DataFrame(flat).to_csv(out / "test_results.csv", index=False)
    pd.concat([pd.DataFrame({"model": n, "path": paths, "session": groups, "label": y_ref, "p_spoiled": p[:, 1]})
               for n, p in preds.items()]).to_csv(out / "predictions.csv", index=False)
    print(f"Wrote {out / 'test_results.json'}, test_results.csv and predictions.csv")


if __name__ == "__main__":
    main()
