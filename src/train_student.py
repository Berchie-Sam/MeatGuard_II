"""
Train, compress and export MeatGuard-S.

    distill   knowledge distillation from the selected teacher (Eq. 5 of the paper), or with --no-kd the
              same recipe with plain cross-entropy (the no-distillation ablation)
    compress  structured channel pruning of the distilled model, then fine-tuning
    export    ONNX and full-integer (INT8) TFLite model for the Android app

Every command writes its results under --out (default runs/student/<step>/): best checkpoint, history.csv
(one row per epoch) and summary.json (configuration, model size and cost, best validation metrics, software
versions and device). Model selection uses the validation split only; test results are produced by
`python -m src.evaluate`.

Usage (from the repository root):
    python -m src.train_student distill  --data data/processed/full/crops
    python -m src.train_student distill  --data data/processed/full/crops --no-kd
    python -m src.train_student compress --data data/processed/full/crops
    python -m src.train_student export   --data data/processed/full/crops
"""
import argparse
import csv
import itertools
import json
import os
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from src.data.dataset import find_split_file, make_loaders
from src.data.transforms import IMG_SIZE, MEAN, STD
from src.model.architecture import MeatGuardS
from src.model.compression import MeatGuardPruner
from src.model.kd_trainer import KDTrainer
from src.train_teacher import check_split, lr_lambda, resolve_device, seed_everything, sha256
from src.utils.metrics import classification_metrics, predict

DISTILL_DEFAULTS = dict(epochs=100, batch_size=32, lr=1e-3, weight_decay=1e-4, warmup_epochs=5, patience=15,
                        alpha=0.3, temperature=4.0, beta=0.1, strategy="domain", seed=42, num_workers=4)
COMPRESS_DEFAULTS = dict(amount=0.3, epochs=10, batch_size=32, lr=1e-4, weight_decay=1e-4,
                         strategy="domain", seed=42, num_workers=4)


# --------------------------------------------------------------------------- shared helpers
def environment(device):
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                "python": platform.python_version()}
    try:
        import torch_pruning as tp
        versions["torch_pruning"] = tp.__version__
    except ImportError:
        pass
    return {"versions": versions,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu"}


def model_cost(model):
    """Parameters (M) and multiply-accumulates (G) for one 224 x 224 image."""
    return MeatGuardPruner(model).stats(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))


def resolve_split(args):
    data_dir = Path(args.data)
    split_file = Path(args.split_file) if args.split_file else find_split_file(data_dir)
    if split_file is None:
        raise FileNotFoundError("No sessions.csv found; pass --split-file (a random split would leak).")
    return data_dir, split_file, check_split(split_file)


def validate(model, loader, device, smoke=False):
    y, p = predict(model, itertools.islice(loader, 3) if smoke else loader, device)
    val = classification_metrics(y, p)
    val["loss"] = float(nn.functional.nll_loss(torch.log(torch.tensor(p).clamp_min(1e-7)), torch.tensor(y)))
    return val


def write_history(path, history):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def better(val, best):
    """Model selection: validation macro-F1, ties broken by lower validation loss."""
    return (val["macro_f1"], -val["loss"]) > (best["macro_f1"], -best["loss"])


def resolve_teacher(path):
    """Teacher checkpoint from teacher_selection.json (written by train_teacher select) or a .pt path."""
    path = Path(path)
    if path.suffix == ".json":
        sel = json.loads(path.read_text())
        ckpt = Path(sel["checkpoint"])
        if not ckpt.exists():                      # runs folder moved (e.g. local -> Drive): look next to the json
            ckpt = path.parent / sel["selected_teacher"] / "best.pt"
        return ckpt, sel["selected_teacher"]
    return path, None


# --------------------------------------------------------------------------- distill
def distill(args):
    cfg = {k: getattr(args, k) for k in DISTILL_DEFAULTS}
    seed_everything(cfg["seed"])
    device = resolve_device(args.device)
    data_dir, split_file, counts = resolve_split(args)
    out = Path(args.out) / ("scratch" if args.no_kd else "distilled")
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, class_weights = make_loaders(
        data_dir, strategy=cfg["strategy"], batch_size=cfg["batch_size"], seed=cfg["seed"],
        split_file=split_file, num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    found = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}
    if any(counts.get(k, 0) != v for k, v in found.items()) and not (args.allow_partial or args.smoke):
        raise ValueError(f"{data_dir} is missing images listed in {split_file.name}: found {found}, expected {counts}")

    student = MeatGuardS(num_classes=2).to(device)
    meta = {"model": "MeatGuardS", "mode": "scratch" if args.no_kd else "distilled",
            "class_names": sorted(p.name for p in data_dir.iterdir() if p.is_dir()),
            "positive_class": "Spoiled_CowMeat", "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
            "split_file": str(split_file), "split_sha256": sha256(split_file), "split_counts": counts,
            "config": cfg, **environment(device), **model_cost(student)}
    student.to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    kd = None
    if args.no_kd:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        teacher_ckpt, teacher_arch = resolve_teacher(args.teacher)
        kd = KDTrainer(student, str(teacher_ckpt), train_loader, val_loader, optimizer, device=device,
                       alpha=cfg["alpha"], temperature=cfg["temperature"], class_weights=class_weights.to(device))
        kd.init_adaptor(next(iter(train_loader))[0][:2])    # before the scheduler, so the adaptor is scheduled too
        meta["teacher"] = {"arch": teacher_arch or getattr(kd.teacher, "arch", None), "checkpoint": str(teacher_ckpt),
                           "params_m": sum(p.numel() for p in kd.teacher.parameters()) / 1e6}
    steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(cfg["warmup_epochs"] * steps, cfg["epochs"] * steps))
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, best, stale, history = 0, {"macro_f1": -1.0, "loss": float("inf")}, 0, []
    last_path = out / "last.pt"
    if last_path.exists() and not args.restart:
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["meta"]["split_sha256"] != meta["split_sha256"]:
            raise ValueError("Checkpoint was trained on a different split file; use --restart.")
        # A resumed run must keep the recipe: e.g. more --epochs would stretch the schedule of a finished run
        # and continue it from its last epoch instead of training from the start
        changed = {k: (v, cfg[k]) for k, v in state["meta"]["config"].items() if k != "num_workers" and cfg.get(k) != v}
        if changed:
            raise ValueError(f"Checkpoint was trained with other settings {changed} (old, new); use --restart "
                             "or another --out.")
        student.load_state_dict(state["model_state_dict"])
        if kd is not None and state.get("adaptor_state_dict"):
            kd.adaptor.load_state_dict(state["adaptor_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale, history = state["epoch"] + 1, state["best"], state["stale"], state["history"]
        print(f"Resuming from epoch {start}")

    print(f"MeatGuard-S ({meta['mode']}): {meta['params_m']:.3f} M params, {meta['gmacs']:.3f} GMACs "
          f"on {meta['device']}" + (f" | teacher {meta['teacher']['arch']}" if kd else ""))
    if stale >= cfg["patience"]:                      # the resumed run had already stopped early
        print("Run already finished (early stopping); nothing to resume.")
    for epoch in range(start, cfg["epochs"] if stale < cfg["patience"] else start):
        student.train()
        t0, sums, seen = time.time(), {"loss": 0.0, "kd": 0.0, "align": 0.0}, 0
        for images, labels in (itertools.islice(train_loader, 3) if args.smoke else train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if kd is None:
                    loss, parts = criterion(student(images), labels), {}
                else:
                    loss, parts = kd.compute_loss(images, labels, beta=cfg["beta"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            n = labels.size(0)
            sums["loss"] += loss.item() * n
            for k, v in parts.items():
                sums[k] += v * n
            seen += n

        val = validate(student, val_loader, device, args.smoke)
        row = {"epoch": epoch, "train_loss": sums["loss"] / seen, "train_kd": sums["kd"] / seen,
               "train_align": sums["align"] / seen, "val_loss": val["loss"], "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "val_recall_spoiled": val["recall_positive"],
               "val_auc": val["roc_auc"], "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        print(f"[{meta['mode']}] epoch {epoch + 1}/{cfg['epochs']}  train {row['train_loss']:.4f}  "
              f"val loss {val['loss']:.4f}  acc {val['accuracy']:.4f}  F1 {val['macro_f1']:.4f}  "
              f"spoiled recall {val['recall_positive']:.4f}  ({row['seconds']:.0f}s)")
        if better(val, best):
            best, stale = {"macro_f1": val["macro_f1"], "loss": val["loss"], "epoch": epoch, "val": val}, 0
            torch.save({"model_state_dict": student.state_dict(), "meta": {**meta, "best_epoch": epoch, "val": val}},
                       out / "best.pt")
        else:
            stale += 1
        torch.save({"model_state_dict": student.state_dict(),
                    "adaptor_state_dict": kd.adaptor.state_dict() if kd is not None and kd.adaptor else None,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "stale": stale,
                    "history": history, "meta": meta}, last_path)
        write_history(out / "history.csv", history)
        if stale >= cfg["patience"]:
            print(f"Early stopping after {cfg['patience']} epochs without improvement.")
            break

    summary = {**meta, "best_epoch": best.get("epoch"), "epochs_run": len(history), "val": best.get("val"),
               "selection_rule": "max validation macro-F1, ties broken by lower validation loss"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best: val macro-F1 {best['macro_f1']:.4f} at epoch {best.get('epoch', -1) + 1} -> {out / 'best.pt'}")


def load_student(path, device="cpu"):
    """A MeatGuard-S checkpoint (best.pt from distill) or a whole pruned model (best_model.pt from compress)."""
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()
    model = MeatGuardS(num_classes=len(obj["meta"]["class_names"]))
    model.load_state_dict(obj["model_state_dict"])
    return model.eval()


# --------------------------------------------------------------------------- compress
def compress(args):
    cfg = {k: getattr(args, k) for k in COMPRESS_DEFAULTS}
    seed_everything(cfg["seed"])
    device = resolve_device(args.device)
    data_dir, split_file, counts = resolve_split(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, _, class_weights = make_loaders(
        data_dir, strategy=cfg["strategy"], batch_size=cfg["batch_size"], seed=cfg["seed"],
        split_file=split_file, num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")

    model = load_student(args.student, device).to(device)
    val_before = validate(model, val_loader, device, args.smoke)
    pruner = MeatGuardPruner(model)
    before, after = pruner.prune_conv_layers(amount=cfg["amount"])
    val_pruned = validate(model, val_loader, device, args.smoke)
    print(f"val macro-F1: distilled {val_before['macro_f1']:.4f} -> pruned, before fine-tuning "
          f"{val_pruned['macro_f1']:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(0, cfg["epochs"] * steps))
    best, history = {"macro_f1": -1.0, "loss": float("inf"), "epoch": None, "val": None}, []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for images, labels in (itertools.islice(train_loader, 3) if args.smoke else train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * labels.size(0)
            seen += labels.size(0)
        val = validate(model, val_loader, device, args.smoke)
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val["loss"],
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "val_recall_spoiled": val["recall_positive"], "val_auc": val["roc_auc"],
                        "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0})
        print(f"[fine-tune] epoch {epoch + 1}/{cfg['epochs']}  train {total / seen:.4f}  val F1 {val['macro_f1']:.4f}  "
              f"spoiled recall {val['recall_positive']:.4f}")
        if better(val, best):
            best = {"macro_f1": val["macro_f1"], "loss": val["loss"], "epoch": epoch, "val": val}
            torch.save(model, out / "best_model.pt")       # whole module: the pruned shapes are not the default ones
        write_history(out / "history.csv", history)

    summary = {"student": str(args.student), "config": cfg, "split_sha256": sha256(split_file),
               "cost_before": before, "cost_after": after, "val_before_pruning": val_before,
               "val_after_pruning_before_finetune": val_pruned, "best_epoch": best["epoch"], "val": best["val"],
               "channels_after": {n: m.out_channels for n, m in model.named_modules() if isinstance(m, nn.Conv2d)},
               "selection_rule": "max validation macro-F1, ties broken by lower validation loss",
               **environment(device)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Pruned model: {after['params_m']:.3f} M params, {after['gmacs']:.3f} GMACs; best val macro-F1 "
          f"{best['macro_f1']:.4f} -> {out / 'best_model.pt'}")


# --------------------------------------------------------------------------- export
def export(args):
    from src.utils.export import export_tflite_int8, tflite_predict
    seed_everything(args.seed)
    data_dir, split_file, _ = resolve_split(args)
    out = Path(args.out)
    # Calibration and the parity check use *training* images without augmentation
    calib_loader, val_loader, _, _ = make_loaders(data_dir, strategy="baseline", batch_size=32, seed=args.seed,
                                                  split_file=split_file, num_workers=args.num_workers)
    model = load_student(args.model, "cpu")
    paths = export_tflite_int8(model, calib_loader, str(out), max_calibration=args.calibration_images)

    # Agreement of the INT8 model with the PyTorch model on validation images
    loader = itertools.islice(val_loader, 3) if args.smoke else val_loader
    y, p_tfl = tflite_predict(paths["tflite_int8"], loader)
    _, p_pt = predict(model, itertools.islice(val_loader, 3) if args.smoke else val_loader, torch.device("cpu"))
    size = lambda f: round(os.path.getsize(f) / 2 ** 20, 3)
    summary = {"model": str(args.model), **paths,
               "size_mb": {"pytorch_fp32": size(args.model), "onnx_fp32": size(paths["onnx"]),
                           "tflite_int8": size(paths["tflite_int8"])},
               "val_int8": classification_metrics(y, p_tfl), "val_fp32": classification_metrics(y, p_pt),
               "val_prediction_agreement": float((p_tfl.argmax(1) == p_pt.argmax(1)).mean()),
               "split_sha256": sha256(split_file), **environment(torch.device("cpu"))}
    try:
        import onnx2tf, tensorflow as tf
        summary["versions"].update({"tensorflow": tf.__version__, "onnx2tf": onnx2tf.__version__})
    except Exception:
        pass
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"INT8 TFLite {summary['size_mb']['tflite_int8']} MB; val macro-F1 FP32 {summary['val_fp32']['macro_f1']:.4f}"
          f" vs INT8 {summary['val_int8']['macro_f1']:.4f}; agreement {summary['val_prediction_agreement']:.4f}")


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    specs = {"distill": (DISTILL_DEFAULTS, "runs/student"), "compress": (COMPRESS_DEFAULTS, "runs/student/pruned"),
             "export": ({"seed": 42, "num_workers": 4}, "runs/student/export")}
    for name, (defaults, out) in specs.items():
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="ImageFolder root of the cropped images")
        p.add_argument("--split-file", default=None, help="sessions.csv (default: next to --data)")
        p.add_argument("--out", default=out)
        p.add_argument("--device", default="auto")
        p.add_argument("--smoke", action="store_true", help="3 train/val batches per epoch, to check the setup")
        for k, v in defaults.items():
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    d = sub.choices["distill"]
    d.add_argument("--teacher", default="runs/teacher/teacher_selection.json",
                   help="teacher_selection.json from train_teacher select, or a teacher best.pt")
    d.add_argument("--no-kd", action="store_true", help="ablation: same recipe with cross-entropy only")
    d.add_argument("--no-amp", action="store_true")
    d.add_argument("--restart", action="store_true", help="ignore last.pt and start over")
    d.add_argument("--allow-partial", action="store_true")
    sub.choices["compress"].add_argument("--student", default="runs/student/distilled/best.pt")
    e = sub.choices["export"]
    e.add_argument("--model", default="runs/student/pruned/best_model.pt")
    e.add_argument("--calibration-images", type=int, default=500)
    args = ap.parse_args()
    {"distill": distill, "compress": compress, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
