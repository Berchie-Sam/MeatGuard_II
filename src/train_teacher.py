"""
Train and select the knowledge-distillation teacher for MeatGuard-S.

Teachers are ImageNet-pretrained torchvision models fine-tuned on the *session split* of the
SAM-cropped MeatScan images. Model selection uses validation macro-F1 only; the test split is
evaluated once, for the selected teacher, by the `select` command.

Usage (from the repository root):
    # 1. train each candidate (resumable; writes runs/teacher/<arch>/)
    python -m src.train_teacher train --arch efficientnet_b0 --data data/processed/full/crops
    python -m src.train_teacher train --arch convnext_tiny   --data data/processed/full/crops
    # 2. pick the best on validation macro-F1, then evaluate it once on test
    python -m src.train_teacher select --data data/processed/full/crops
    # lightweight baselines use the same recipe in their own folder (never candidates for selection)
    python -m src.train_teacher train --arch mobilenet_v3_small --data data/processed/full/crops --out runs/baselines

The split file defaults to <data>/../sessions.csv (see src/data/sessions.py).
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from src.data.dataset import find_split_file, make_loaders
from src.data.transforms import IMG_SIZE, MEAN, STD
from src.model.teacher import ALL_ARCHS, build_teacher, load_teacher
from src.utils.metrics import bootstrap_ci, classification_metrics, predict

DEFAULTS = dict(epochs=30, batch_size=64, lr=3e-4, weight_decay=0.05, warmup_epochs=2,
                label_smoothing=0.1, patience=8, strategy="domain", seed=42, num_workers=4)


# --------------------------------------------------------------------------- utilities
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_split(split_file):
    """Every capture session must sit in exactly one partition; returns the per-split counts."""
    table = pd.read_csv(split_file)
    spread = table.groupby("session")["split"].nunique()
    if (spread > 1).any():
        raise ValueError(f"{int((spread > 1).sum())} sessions span several partitions in {split_file}")
    return table["split"].value_counts().to_dict()


def session_groups(dataset, data_dir, split_file):
    """Session id for every sample of a dataset, in loader order (for session-level bootstrap CIs)."""
    table = pd.read_csv(split_file)
    key = lambda rel: Path(rel).with_suffix("").as_posix()
    session_of = dict(zip(table["path"].map(key), table["session"]))
    return [session_of[key(Path(p).relative_to(data_dir))] for p, _ in dataset.samples]


def lr_lambda(warmup_steps, total_steps):
    """Linear warmup followed by cosine decay to zero."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- training
def train(args):
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    seed_everything(cfg["seed"])
    device = resolve_device(args.device)
    data_dir = Path(args.data)
    split_file = Path(args.split_file) if args.split_file else find_split_file(data_dir)
    if split_file is None:
        raise FileNotFoundError("No sessions.csv found; pass --split-file (a random split would leak).")
    counts = check_split(split_file)

    out = Path(args.out) / args.arch
    out.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, _, class_weights = make_loaders(
        data_dir, strategy=cfg["strategy"], batch_size=cfg["batch_size"], seed=cfg["seed"],
        split_file=split_file, num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    found = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}
    missing = {k: counts.get(k, 0) - v for k, v in found.items() if counts.get(k, 0) != v}
    if missing and not (args.allow_partial or args.smoke):
        raise ValueError(f"{data_dir} is missing images listed in {split_file.name}: {missing}. "
                         "Finish cropping/copying first, or pass --allow-partial.")

    model = build_teacher(args.arch, num_classes=len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(cfg["warmup_epochs"] * steps_per_epoch, cfg["epochs"] * steps_per_epoch))
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    meta = {
        "arch": args.arch, "class_names": class_names, "positive_class": "Spoiled_CowMeat",
        "img_size": IMG_SIZE, "mean": MEAN, "std": STD, "data_dir": str(data_dir),
        "split_file": str(split_file), "split_sha256": sha256(split_file), "split_counts": counts,
        "config": cfg, "pretrained": not args.no_pretrained,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "python": platform.python_version()},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }

    start_epoch, best, stale, history = 0, {"macro_f1": -1.0, "loss": float("inf")}, 0, []
    last_path = out / "last.pt"
    if last_path.exists() and not args.restart:          # resume an interrupted run
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["meta"]["split_sha256"] != meta["split_sha256"]:
            raise ValueError("Checkpoint was trained on a different split file; use --restart.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, stale, history = state["epoch"] + 1, state["best"], state["stale"], state["history"]
        print(f"Resuming {args.arch} from epoch {start_epoch}")

    print(f"{args.arch}: {meta['params_m']:.2f} M params on {meta['device']} | split {counts}")
    if stale >= cfg["patience"]:                      # the resumed run had already stopped early
        print("Run already finished (early stopping); nothing to resume.")
    for epoch in range(start_epoch, cfg["epochs"] if stale < cfg["patience"] else start_epoch):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        batches = itertools.islice(train_loader, 3) if args.smoke else train_loader
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)

        y_val, p_val = predict(model, itertools.islice(val_loader, 3) if args.smoke else val_loader, device)
        val_loss = float(nn.functional.cross_entropy(torch.log(torch.tensor(p_val).clamp_min(1e-7)),
                                                     torch.tensor(y_val), reduction="mean"))
        val = classification_metrics(y_val, p_val)
        row = {"epoch": epoch, "train_loss": running / seen, "val_loss": val_loss, "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "val_recall_spoiled": val["recall_positive"],
               "val_auc": val["roc_auc"], "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        print(f"[{args.arch}] epoch {epoch + 1}/{cfg['epochs']}  train {row['train_loss']:.4f}  "
              f"val loss {val_loss:.4f}  acc {val['accuracy']:.4f}  F1 {val['macro_f1']:.4f}  "
              f"spoiled recall {val['recall_positive']:.4f}  ({row['seconds']:.0f}s)")

        improved = (val["macro_f1"], -val_loss) > (best["macro_f1"], -best["loss"])
        if improved:
            best, stale = {"macro_f1": val["macro_f1"], "loss": val_loss, "epoch": epoch, "val": val}, 0
            torch.save({"model_state_dict": model.state_dict(), "meta": {**meta, "best_epoch": epoch, "val": val}},
                       out / "best.pt")
        else:
            stale += 1
        torch.save({"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best": best, "stale": stale, "history": history, "meta": meta}, last_path)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        if stale >= cfg["patience"]:
            print(f"Early stopping after {cfg['patience']} epochs without improvement.")
            break

    summary = {"arch": args.arch, "params_m": meta["params_m"], "best_epoch": best.get("epoch"),
               "val": best.get("val"), "split_sha256": meta["split_sha256"], "config": cfg}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best {args.arch}: val macro-F1 {best['macro_f1']:.4f} at epoch {best.get('epoch', -1) + 1}")


# --------------------------------------------------------------------------- selection
def select(args):
    device = resolve_device(args.device)
    runs = Path(args.out)
    summaries = [json.loads(p.read_text()) for p in sorted(runs.glob("*/summary.json"))]
    if not summaries:
        raise FileNotFoundError(f"No trained teachers under {runs}")
    shas = {s["split_sha256"] for s in summaries}
    if len(shas) > 1:
        raise ValueError("Candidates were trained on different split files; retrain them on one split.")

    # Rank on validation only: macro-F1, then spoiled-class recall, then fewer parameters
    ranked = sorted(summaries, key=lambda s: (-s["val"]["macro_f1"], -s["val"]["recall_positive"], s["params_m"]))
    print("Validation ranking (test not used):")
    for s in ranked:
        print(f"  {s['arch']:16} macro-F1 {s['val']['macro_f1']:.4f}  spoiled recall "
              f"{s['val']['recall_positive']:.4f}  acc {s['val']['accuracy']:.4f}  {s['params_m']:.1f} M")
    winner = ranked[0]["arch"]

    model, meta = load_teacher(runs / winner / "best.pt", map_location=device)
    model.to(device)
    data_dir = Path(args.data)
    split_file = Path(meta["split_file"]) if Path(meta["split_file"]).exists() else find_split_file(data_dir)
    if sha256(split_file) != meta["split_sha256"]:
        raise ValueError("Split file differs from the one used in training.")
    _, _, test_loader, _ = make_loaders(data_dir, strategy="baseline", batch_size=64, split_file=split_file,
                                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    y, p = predict(model, test_loader, device)
    test = classification_metrics(y, p)
    groups = session_groups(test_loader.dataset, data_dir, split_file)
    test["macro_f1_ci95_sessions"] = bootstrap_ci(y, p, groups, "macro_f1")
    test["recall_positive_ci95_sessions"] = bootstrap_ci(y, p, groups, "recall_positive")
    result = {"selected_teacher": winner, "checkpoint": str(runs / winner / "best.pt"),
              "selection_rule": "max validation macro-F1, then spoiled recall, then fewer parameters",
              "validation_ranking": [{"arch": s["arch"], **{k: s["val"][k] for k in
                                      ("macro_f1", "recall_positive", "accuracy", "roc_auc")}} for s in ranked],
              "test": test, "n_test_images": int(len(y)), "n_test_sessions": int(len(set(groups)))}
    (runs / "teacher_selection.json").write_text(json.dumps(result, indent=2))
    print(f"Selected {winner}: test macro-F1 {test['macro_f1']:.4f} "
          f"(95% CI over sessions {test['macro_f1_ci95_sessions'][0]:.3f}–{test['macro_f1_ci95_sessions'][1]:.3f}), "
          f"spoiled recall {test['recall_positive']:.4f}, accuracy {test['accuracy']:.4f}")
    return result


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "select"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="ImageFolder root of the cropped images")
        p.add_argument("--split-file", default=None, help="sessions.csv (default: next to --data)")
        p.add_argument("--out", default="runs/teacher")
        p.add_argument("--device", default="auto")
        p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    t = sub.choices["train"]
    t.add_argument("--arch", required=True, choices=sorted(ALL_ARCHS))
    for k, v in DEFAULTS.items():
        if k != "num_workers":
            t.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    t.add_argument("--no-pretrained", action="store_true", help="train from scratch (ablation only)")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--restart", action="store_true", help="ignore last.pt and start over")
    t.add_argument("--smoke", action="store_true", help="3 train/val batches per epoch, to check the setup")
    t.add_argument("--allow-partial", action="store_true", help="train even if some split images are missing")
    args = ap.parse_args()
    if args.command == "train":
        train(args)
    else:
        select(args)


if __name__ == "__main__":
    main()
