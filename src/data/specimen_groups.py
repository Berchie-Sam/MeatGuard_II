"""
Specimen-level split: link capture sessions that show the same piece of meat, then assign whole groups to
train/val/test.

The capture sessions of sessions.py link near-identical frames by a perceptual hash; they split continuous capture
runs and miss the same piece photographed at another zoom or angle, or mirrored (src.data.specimen_check found a
same-class train match for 74% of test images). Here sessions are linked into specimen groups:

1. Candidates: for every image, the k most similar images of the same class in *other* sessions, by cosine
   similarity of DINOv2 embeddings (either orientation of the image; embeddings.npz from specimen_check).
2. Verification: candidate pairs are processed from the most to the least similar; a pair whose sessions are not
   yet in the same group is verified with AKAZE + RANSAC (both orientations), and the two sessions are linked when
   the number of inliers reaches --min-inliers. Pairs of already linked sessions are skipped.
3. Groups are the connected components of the session graph; they are assigned to splits per class with the rule
   of sessions.py (largest first, to the split furthest below its target; groups above --max-holdout-frac of a
   held-out target go to train).
4. Residual check: each val/test image against its nearest same-class train images (and an equal-sized
   other-class pool as the chance rate), verified the same way.

Writes --out (default <split-file dir>/specimens.csv) with columns path, class, session (= specimen group, so all
training and evaluation code groups by specimen), capture_session, split; and edges.csv, groups_report.json next
to it.

Usage:
    python -m src.data.specimen_groups --data data/processed/full/crops \
        --split-file data/processed/full/sessions.csv --embeddings runs/specimen_check/embeddings.npz
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sessions import SPLITS, _UnionFind, assign_splits
from src.data.specimen_check import Verifier


def candidates(table, emb, emb_flip, k, min_cos, chunk=512):
    """Same-class pairs (i, j, cosine) with j among the k most similar images of i outside i's session."""
    pairs = {}
    session = table["session"].to_numpy()
    for cls, sub in table.groupby("class"):
        idx = sub.index.to_numpy()
        e = emb[idx]
        for s in range(0, len(idx), chunk):
            q = idx[s:s + chunk]
            sim = np.maximum(emb[q] @ e.T, emb_flip[q] @ e.T)
            sim[session[q][:, None] == session[idx][None, :]] = -1          # other sessions only
            top = np.argpartition(-sim, k, axis=1)[:, :k]
            for r, i in enumerate(q):
                for c in top[r]:
                    j, cos = idx[c], float(sim[r, c])
                    if cos >= min_cos:
                        key = (min(i, j), max(i, j))
                        pairs[key] = max(pairs.get(key, -1), cos)
    out = pd.DataFrame([(a, b, c) for (a, b), c in pairs.items()], columns=["i", "j", "cosine"])
    return out.sort_values("cosine", ascending=False, ignore_index=True)


def link_sessions(table, pairs, verifier, min_inliers, edges_path, resume=False):
    """Union-find over sessions; verify a pair only if its sessions are not linked yet. With `resume`, pairs
    already verified in an existing edges file are taken from it instead of being verified again."""
    sessions = table["session"].to_numpy()
    ids = {s: n for n, s in enumerate(np.unique(sessions))}
    uf = _UnionFind(len(ids))
    path = table["path"].to_numpy()
    known = {}
    previous = Path(edges_path).with_name("edges_previous.csv")
    if resume and previous.exists():
        prev = pd.read_csv(previous)
        known = {(a, b): (int(n), bool(f)) for a, b, n, f in zip(prev["a"], prev["b"], prev["inliers"], prev["flipped"])}
        print(f"resuming: {len(known)} pairs already verified in {previous}")
    done = links = 0
    t0 = time.time()
    with open(edges_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "session_a", "session_b", "cosine", "inliers", "flipped", "linked"])
        for n, (i, j, cos) in enumerate(pairs.itertuples(index=False)):
            sa, sb = ids[sessions[i]], ids[sessions[j]]
            if uf.find(sa) == uf.find(sb):
                continue
            inl, flipped = known.get((path[i], path[j])) or verifier.best(path[i], path[j])
            done += 1
            linked = inl >= min_inliers
            if linked:
                uf.union(sa, sb)
                links += 1
            w.writerow([path[i], path[j], sessions[i], sessions[j], round(cos, 4), inl, flipped, linked])
            if done % 500 == 0:
                f.flush()
                print(f"  verified {done} pairs ({n + 1}/{len(pairs)} candidates), {links} links, "
                      f"{(time.time() - t0) / done * 1000:.0f} ms/pair", flush=True)
    print(f"verified {done} pairs, {links} session links")
    root = {s: uf.find(n) for s, n in ids.items()}
    return pd.Series(sessions).map(root).pipe(lambda r: pd.factorize(r)[0])


def residual_check(table, emb, emb_flip, verifier, k, rng, chance=True, min_inliers=20):
    """Per val/test image: max inliers against its k nearest same-class train images and, as the chance rate,
    against an equal-sized other-class train pool. Also returns the same-class pairs reaching `min_inliers`."""
    out, found = {}, []
    train = table.index[table["split"] == "train"].to_numpy()
    tcls = table.loc[train, "class"].to_numpy()
    for name in ("val", "test"):
        for pool_name in ("same_class", "other_class") if chance else ("same_class",):
            best = []
            for cls in sorted(table["class"].unique()):
                q = table.index[(table["split"] == name) & (table["class"] == cls)].to_numpy()
                pool = train[tcls == cls] if pool_name == "same_class" else train[tcls != cls]
                if pool_name == "other_class":
                    pool = rng.choice(pool, min(len(pool), (tcls == cls).sum()), replace=False)
                sim = np.maximum(emb[q] @ emb[pool].T, emb_flip[q] @ emb[pool].T)
                top = np.argsort(-sim, axis=1)[:, :k]
                for r, i in enumerate(q):
                    res = [(verifier.best(table.at[i, "path"], table.at[pool[c], "path"])[0], pool[c])
                           for c in top[r]]
                    inl, j = max(res)
                    best.append(inl)
                    if pool_name == "same_class" and inl >= min_inliers:
                        found.append((i, j, inl))
            best = np.array(best)
            out[f"{name}|{pool_name}"] = {"n": int(len(best)),
                                          **{f"inliers>={t}": int((best >= t).sum()) for t in (15, 20, 30, 50)}}
            print(f"  residual {name} {pool_name}: {out[f'{name}|{pool_name}']}", flush=True)
    return out, found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="crops/ ImageFolder root")
    ap.add_argument("--split-file", required=True, help="capture-session split (sessions.csv)")
    ap.add_argument("--embeddings", required=True, help="embeddings.npz from src.data.specimen_check")
    ap.add_argument("--out", default=None, help="default: specimens.csv next to --split-file")
    ap.add_argument("--k", type=int, default=6, help="candidate neighbours per image")
    ap.add_argument("--min-cos", type=float, default=0.7)
    ap.add_argument("--min-inliers", type=int, default=20)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--max-holdout-frac", type=float, default=0.25)
    ap.add_argument("--residual-k", type=int, default=3)
    ap.add_argument("--refine-rounds", type=int, default=4,
                    help="link residual val/test-train matches and re-split, up to this many times")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="reuse pairs already verified in edges.csv")
    ap.add_argument("--cache", type=int, default=500, help="AKAZE feature sets kept in memory")
    args = ap.parse_args()

    table = pd.read_csv(args.split_file).reset_index(drop=True)
    npz = np.load(args.embeddings)
    emb, emb_flip = npz["emb"].astype(np.float32), npz["emb_flip"].astype(np.float32)
    assert len(emb) == len(table), "embeddings do not match the split file (run specimen_check on this file)"
    out = Path(args.out) if args.out else Path(args.split_file).with_name("specimens.csv")
    verifier = Verifier(Path(args.data), max_cache=args.cache)

    pairs = candidates(table, emb, emb_flip, args.k, args.min_cos)
    print(f"{len(table)} images, {table['session'].nunique()} capture sessions; {len(pairs)} candidate pairs")
    edges = out.with_name("edges.csv")
    if args.resume and edges.exists():                      # keep the previous file; the new one is rewritten
        edges.replace(edges.with_name("edges_previous.csv"))
    groups = link_sessions(table, pairs, verifier, args.min_inliers, edges, resume=args.resume)

    spec = table.rename(columns={"session": "capture_session"}).drop(columns="split")
    spec["session"] = groups
    ratios = dict(zip(SPLITS, args.ratios))
    spec["split"] = assign_splits(spec, ratios, args.seed, args.max_holdout_frac)

    # Refinement: residual matches are links the candidate search missed; merge their groups and re-split
    rng = np.random.default_rng(args.seed)
    refine_log = []
    for rnd in range(1, args.refine_rounds + 1):
        print(f"refinement round {rnd}:")
        _, found = residual_check(spec, emb, emb_flip, verifier, args.residual_k, rng, chance=False,
                                  min_inliers=args.min_inliers)
        refine_log.append({"round": rnd, "residual_links": len(found)})
        if not found:
            break
        uf = _UnionFind(int(spec["session"].max()) + 1)
        for i, j, _ in found:
            uf.union(int(spec.at[i, "session"]), int(spec.at[j, "session"]))
        spec["session"] = pd.factorize(spec["session"].map(uf.find))[0]
        spec["split"] = assign_splits(spec, ratios, args.seed, args.max_holdout_frac)
        print(f"  merged {len(found)} residual links; {spec['session'].nunique()} groups")
    sizes = spec.groupby("session").size()
    per = spec.groupby("session")["capture_session"].nunique()
    print(f"specimen groups: {len(sizes)} | largest {sizes.max()} images | "
          f"groups spanning >1 capture session: {(per > 1).sum()}")
    print(pd.crosstab(spec["split"], spec["class"], margins=True))
    spec[["path", "class", "session", "capture_session", "split"]].to_csv(out, index=False)
    print("wrote", out)

    print("final residual check (with chance pool):")
    residual, _ = residual_check(spec, emb, emb_flip, verifier, args.residual_k, rng,
                                 min_inliers=args.min_inliers)
    report = {"refinement": refine_log, "settings": vars(args), "n_candidate_pairs": int(len(pairs)),
              "n_groups": int(len(sizes)), "largest_group": int(sizes.max()),
              "groups_spanning_sessions": int((per > 1).sum()),
              "group_sizes_by_class": {c: spec[spec["class"] == c].groupby("session").size().describe().round(1)
                                       .to_dict() for c in spec["class"].unique()},
              "counts": pd.crosstab(spec["split"], spec["class"]).to_dict(),
              "groups_per_split": spec.groupby("split")["session"].nunique().to_dict(),
              "residual_leakage": residual}
    out.with_name("groups_report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
