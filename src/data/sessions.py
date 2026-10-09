"""
Session-aware (group) split for MeatScan.

MeatScan images are burst/video frames: consecutively numbered files are often
near-identical shots of the same piece of meat. A per-image random split puts
copies of the same scene in train and test (~68% of test images had a train
near-duplicate). Here images are grouped into capture sessions and whole
sessions are assigned to train/val/test, stratified by class.

Session linking (perceptual dHash, minimum over the 8 rotations/flips):
  * frames of the same class within --window positions in file-number order with
    distance <= --adj-thresh
  * any two same-class images with distance <= --dup-thresh (rotated/flipped copies)
Thresholds were calibrated against cross-class pairs (which cannot be the same
scene): chance link rate ~0.06% at 12 and ~0% at 6 on a 360k-pair sample.

Usage:
    python -m src.data.sessions --src E:/Datasets/MeatScan --out E:/Datasets/full/sessions.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_TRANSPOSES = [None, Image.FLIP_LEFT_RIGHT]


def dhash_variants(path):
    """64-bit dHash for each of the 8 rotations/flips, packed to 8 bytes each."""
    gray = Image.open(path).convert("L")
    out = []
    for k in range(4):
        rot = gray.rotate(90 * k)
        for t in _TRANSPOSES:
            v = rot if t is None else rot.transpose(t)
            a = np.asarray(v.resize((9, 8), Image.BILINEAR), dtype=np.int16)
            out.append(np.packbits((a[:, 1:] > a[:, :-1]).ravel()))
    return np.stack(out)  # (8, 8) uint8


def min_distance(hashes, i, js):
    """Min Hamming distance between image i (any orientation) and images js (identity)."""
    x = hashes[i][:, None, :] ^ hashes[js, 0, :][None]
    return _POPCOUNT[x].sum(2).min(0)


def file_number(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def build_sessions(df, hashes, adj_thresh, dup_thresh, window, chunk=200):
    uf = _UnionFind(len(df))
    for cls, sub in df.groupby("class"):
        idx = sub.sort_values("number").index.to_numpy()
        # Near-in-sequence frames
        for k, i in enumerate(idx[:-1]):
            nbrs = idx[k + 1:k + 1 + window]
            for j in nbrs[min_distance(hashes, i, nbrs) <= adj_thresh]:
                uf.union(i, j)
        # Near-identical copies anywhere in the class
        for s in range(0, len(idx), chunk):
            block = idx[s:s + chunk]
            x = hashes[block][:, :, None, :] ^ hashes[idx, 0, :][None, None]
            d = _POPCOUNT[x].sum(3).min(1)
            for a, b in zip(*np.nonzero(d <= dup_thresh)):
                if block[a] != idx[b]:
                    uf.union(block[a], idx[b])
    roots = np.array([uf.find(i) for i in range(len(df))])
    return pd.factorize(roots)[0]


def assign_splits(df, ratios, seed, max_holdout_frac):
    """Per class, assign whole sessions (largest first) to the split furthest below its target.

    Sessions larger than `max_holdout_frac` of the smallest held-out target go to
    train, so val/test are built from many sessions instead of one dominant scene.
    """
    rng = np.random.default_rng(seed)
    split_of = {}
    for cls, sub in df.groupby("class"):
        sizes = sub.groupby("session").size()
        order = sizes.index.to_numpy().copy()  # copy: shuffling the index view corrupts `sizes`
        rng.shuffle(order)  # random tie-break among equal sizes
        order = sorted(order, key=lambda s: -sizes[s])
        total, filled = len(sub), dict.fromkeys(SPLITS, 0)
        cap = max_holdout_frac * min(ratios["val"], ratios["test"]) * total
        for s in order:
            deficit = {k: ratios[k] * total - filled[k] for k in SPLITS}
            target = "train" if sizes[s] > cap else \
                max(SPLITS, key=lambda k: deficit[k] / max(ratios[k], 1e-9))
            split_of[s] = target
            filled[target] += sizes[s]
    return df["session"].map(split_of)


def leakage_report(df, hashes, thresh, seed=0):
    """Share of val/test images with a train image within `thresh`.

    `same` uses same-class train images; `chance` uses an equally sized pool of
    other-class train images, which cannot be the same scene, so it estimates
    the rate expected from hash collisions alone.
    """
    rng = np.random.default_rng(seed)
    train = df.index[df.split == "train"].to_numpy()
    train_cls = df.loc[train, "class"].to_numpy()
    out = {}
    for name in ("val", "test"):
        idx = df.index[df.split == name].to_numpy()
        same = chance = 0
        for i in idx:
            pool = train[train_cls == df.at[i, "class"]]
            other = train[train_cls != df.at[i, "class"]]
            other = rng.choice(other, min(len(pool), len(other)), replace=False)
            same += int(min_distance(hashes, i, pool).min() <= thresh)
            chance += int(min_distance(hashes, i, other).min() <= thresh)
        n = max(len(idx), 1)
        out[name] = (same / n, chance / n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adj-thresh", type=int, default=12)
    ap.add_argument("--dup-thresh", type=int, default=6)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--max-holdout-frac", type=float, default=0.25,
                    help="sessions above this fraction of a val/test class target are forced into train")
    args = ap.parse_args()

    src = Path(args.src)
    paths = sorted(p for p in src.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.parent != src)
    df = pd.DataFrame({
        "path": [p.relative_to(src).as_posix() for p in paths],
        "class": [p.parent.name for p in paths],
        "number": [file_number(p.stem) for p in paths],
    })
    print(f"hashing {len(df)} images ...", flush=True)
    hashes = np.stack([dhash_variants(p) for p in paths])

    df["session"] = build_sessions(df, hashes, args.adj_thresh, args.dup_thresh, args.window)
    df["split"] = assign_splits(df, dict(zip(SPLITS, args.ratios)), args.seed, args.max_holdout_frac)

    sizes = df.groupby("session").size()
    print(f"sessions: {len(sizes)} | singletons: {(sizes == 1).sum()} | "
          f"largest: {sizes.max()} | median: {sizes.median():.0f}")
    print(pd.crosstab(df["split"], df["class"], margins=True))
    print("val/test images with a train near-duplicate (same-class vs chance baseline):")
    for t in (4, 6, 8, 10):
        rep = leakage_report(df, hashes, t)
        print(f"  d<={t:>2}: " + "  ".join(
            f"{k} {same:.1%} (chance {chance:.1%})" for k, (same, chance) in rep.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns="number").to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
