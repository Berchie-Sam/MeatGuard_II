"""
Config-driven entry point for the MeatGuard II pipeline.

Reads configs/config.yaml (and the data / model YAMLs it points to) and runs one step, or all of them, by calling
the step scripts with the configured settings. Each step writes its results under `runs_dir`; training steps
resume from their last checkpoint when re-run, so `--mode all` can be repeated after an interruption.

Steps (in pipeline order):
    sessions   capture sessions and the session-level split          src/data/sessions.py
    segment    meat masks and crops (Grounding DINO + SAM 2.1)       src/data/segment_meat.py
    teacher    train the teacher candidates, select one on validation src/train_teacher.py
    baselines  lightweight comparison models (teacher recipe)        src/train_teacher.py --out <runs>/baselines
    scratch    MeatGuard-S without distillation (ablation)           src/train_student.py distill --no-kd
    distill    MeatGuard-S distilled from the selected teacher       src/train_student.py distill
    compress   structured pruning + fine-tuning                      src/train_student.py compress
    export     ONNX + INT8 TFLite                                    src/train_student.py export
    evaluate   test evaluation of all trained models                 src/evaluate.py
    xai        evaluation of the explanations                        src/evaluate_xai.py
    collect    zip all result files (no weights)                     src/collect_results.py
`all` runs teacher ... collect (the data steps are run once, separately).

Usage (from the repository root):
    python -m src.main --mode distill
    python -m src.main --mode all --runs runs_colab
    python -m src.main --mode all --smoke          # three batches per step, results in <runs>_smoke/
    python -m src.main --mode teacher --dry-run    # print the commands only
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config_manager import ConfigManager  # noqa: E402

STEPS = ["sessions", "segment", "teacher", "baselines", "scratch", "distill", "compress", "export",
         "evaluate", "xai", "collect"]
ALL = STEPS[2:]


def flags(**kw):
    """--key value pairs (underscores -> dashes); True adds a bare flag, None/False are skipped."""
    out = []
    for k, v in kw.items():
        if v is None or v is False:
            continue
        out.append(f"--{k.replace('_', '-')}")
        if v is not True:
            out.append(str(v))
    return out


def reuse_teacher_run(runs, arch):
    """A baseline that was also a teacher candidate (same recipe and seed) is copied, not retrained."""
    src, dst = runs / "teacher" / arch, runs / "baselines" / arch
    if (dst / "summary.json").exists():
        return False                                    # train_teacher resumes / finishes it as usual
    if (src / "summary.json").exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"[baselines] reusing the teacher run of {arch}")
        return True
    return False


def commands(step, cfg, runs, smoke):
    """The `python -m ...` argument lists for one step."""
    data = cfg["data_details"]
    common = flags(data=data["path"], split_file=data.get("split_file"), device=cfg.get("device", "auto"))
    workers = flags(num_workers=cfg.get("num_workers", 4))
    seed = flags(seed=cfg.get("seed", 42))
    smoke_flag = flags(smoke=smoke)
    t, s = dict(cfg.get("teacher", {})), dict(cfg.get("training", {}))
    kd, comp, xai = cfg.get("distillation", {}), dict(cfg.get("compression", {})), cfg.get("xai", {})
    if smoke:                                           # one short epoch per training step
        t["epochs"] = s["epochs"] = comp["epochs"] = 1
    teacher_args = flags(epochs=t.get("epochs"), batch_size=t.get("batch_size"), lr=t.get("lr"),
                         weight_decay=t.get("weight_decay"), warmup_epochs=t.get("warmup_epochs"),
                         label_smoothing=t.get("label_smoothing"), patience=t.get("patience"),
                         strategy=cfg.get("augment"))
    student_args = flags(epochs=s.get("epochs"), batch_size=s.get("batch_size"), lr=s.get("lr"),
                         weight_decay=s.get("weight_decay"), warmup_epochs=s.get("warmup_epochs"),
                         patience=s.get("patience"), strategy=cfg.get("augment"))
    if step == "sessions":
        return [["src.data.sessions", *flags(src=data["source"], out=data["split_file"], seed=cfg.get("seed", 42))]]
    if step == "segment":
        return [["src.data.segment_meat", *flags(src=data["source"], out=data["root"])]]
    if step == "teacher":
        out = flags(out=runs / "teacher")
        return ([["src.train_teacher", "train", *flags(arch=a), *common, *out, *workers, *seed, *teacher_args,
                  *smoke_flag] for a in t.get("archs", [])] +
                [["src.train_teacher", "select", *common, *out, *workers]])
    if step == "baselines":
        archs = [a for a in cfg.get("baselines", []) if not reuse_teacher_run(runs, a)]
        return [["src.train_teacher", "train", *flags(arch=a), *common, *flags(out=runs / "baselines"), *workers,
                 *seed, *teacher_args, *smoke_flag] for a in archs]
    if step in ("scratch", "distill"):
        extra = (flags(no_kd=True) if step == "scratch" else
                 flags(teacher=runs / "teacher" / "teacher_selection.json", alpha=kd.get("alpha"),
                       temperature=kd.get("temperature"), beta=kd.get("beta")))
        return [["src.train_student", "distill", *common, *flags(out=runs / "student"), *workers, *seed,
                 *student_args, *extra, *smoke_flag]]
    if step == "compress":
        return [["src.train_student", "compress", *common, *workers, *seed, *smoke_flag,
                 *flags(student=runs / "student" / "distilled" / "best.pt", out=runs / "student" / "pruned",
                        amount=comp.get("amount"), epochs=comp.get("epochs"), lr=comp.get("lr"),
                        weight_decay=comp.get("weight_decay"), batch_size=s.get("batch_size"),
                        strategy=cfg.get("augment"))]]
    if step == "export":
        return [["src.train_student", "export", *common, *workers, *seed, *smoke_flag,
                 *flags(model=runs / "student" / "pruned" / "best_model.pt", out=runs / "student" / "export",
                        calibration_images=cfg.get("export", {}).get("calibration_images"))]]
    if step == "evaluate":
        return [["src.evaluate", *common, *workers, *flags(runs=runs)]]
    if step == "xai":
        quick = dict(val_per_class=2, test_per_class=2, shap_background=8, shap_samples=10, sanity_images=1,
                     random_repeats=1) if smoke else {}
        settings = {k: xai.get(k) for k in ("val_per_class", "test_per_class", "per_session", "shap_background",
                                             "shap_samples", "sanity_images")}
        return [["src.evaluate_xai", *common, *seed,
                 *flags(masks=data.get("masks"), model=runs / "student" / "pruned" / "best_model.pt",
                        out=runs / "xai", **{**settings, **quick})]]
    if step == "collect":
        return [["src.collect_results", *flags(runs=runs, out=runs / "results_bundle.zip")]]
    raise ValueError(f"Unknown mode '{step}'. Choose from {STEPS + ['all']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--mode", help="override the config's mode: " + ", ".join(STEPS + ["all"]))
    ap.add_argument("--runs", help="override runs_dir")
    ap.add_argument("--smoke", action="store_true", help="three batches per step; results in <runs>_smoke/")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    cfg = ConfigManager(args.config).config
    mode = args.mode or cfg.get("mode", "all")
    runs = Path(args.runs or cfg.get("runs_dir", "runs"))
    if args.smoke:
        runs = runs.with_name(runs.name + "_smoke")
    steps = ALL if mode == "all" else [mode]
    for step in steps:
        for cmd in commands(step, cfg, runs, args.smoke):
            full = [sys.executable, "-m", *map(str, cmd)]
            print(f"\n=== [{step}] {' '.join(full[1:])}", flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(full, cwd=ROOT)
            if result.returncode != 0:
                sys.exit(f"Step '{step}' failed (exit code {result.returncode}).")
    print(f"\nDone: {', '.join(steps)}. Results in {runs}/")


if __name__ == "__main__":
    main()
