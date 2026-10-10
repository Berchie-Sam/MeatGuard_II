"""
Bundle the result files under runs/ (JSON, CSV, example maps; no model weights) into one zip to share.

Usage:  python -m src.collect_results --runs runs --out results_bundle.zip
"""
import argparse
import zipfile
from pathlib import Path

KEEP = {".json", ".csv", ".npz", ".png", ".txt"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results_bundle.zip")
    args = ap.parse_args()
    runs = Path(args.runs)
    files = sorted(p for p in runs.rglob("*") if p.is_file() and p.suffix in KEEP)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path("runs") / p.relative_to(runs))
    print(f"{len(files)} files -> {args.out}")
    for p in files:
        print("  ", p.relative_to(runs))


if __name__ == "__main__":
    main()
