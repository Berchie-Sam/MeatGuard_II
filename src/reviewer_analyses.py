"""Analyses added in response to the simulated pre-submission review (2026-11-15).

1. Cluster-aware paired comparisons on the specimen-level split. Images of one specimen group are not independent, so
   the exact McNemar test on images (Table 6) overstates significance. Reported here for each pair of models:
   - the accuracy difference with a 95% bootstrap CI from resampling test specimen groups;
   - a two-sided bootstrap p-value (share of resamples on the other side of zero, doubled);
   - the cluster-adjusted McNemar test of Durkalski et al. (2003), Stat Med 22(15):2417-2428,
     X2 = (sum_k (b_k - c_k))^2 / sum_k (b_k - c_k)^2, 1 df, with b_k, c_k the discordant counts in group k.
2. Leakage subset analysis on the capture-session split. Test images are divided into those with a same-class training
   image of the same piece of meat (>= 30 inliers among the three most similar training images, as in Section 3.2.3)
   and those without; every model trained on the session split is evaluated on both subsets.

Usage (paths default to the paper folder layout):
    python -m src.reviewer_analyses --paper-dir "D:/Research_Work/paperpwork/AI/MeatGuard_II-paper"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

N_BOOT = 10000
SEED = 0


def metrics(y, p, thr=0.5):
    pred = (p > thr).astype(int)  # ties at 0.5 -> fresh, as argmax in evaluate.py
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    f1_s = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    f1_f = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else np.nan
    return {"n": int(len(y)), "n_fresh": int((y == 0).sum()), "n_spoiled": int((y == 1).sum()),
            "accuracy": (tp + tn) / len(y), "macro_f1": np.nanmean([f1_s, f1_f]),
            "spoiled_recall": tp / (tp + fn) if tp + fn else np.nan,
            "fresh_recall": tn / (tn + fp) if tn + fp else np.nan,
            "balanced_accuracy": np.nanmean([tp / (tp + fn) if tp + fn else np.nan, tn / (tn + fp) if tn + fp else np.nan]),
            "errors": fp + fn}


def group_index(groups):
    uniq = np.unique(groups)
    return uniq, [np.flatnonzero(groups == g) for g in uniq]


def paired_comparison(y, pa, pb, groups, rng):
    """A = reference model, B = compared model. Positive difference = B more accurate."""
    ok_a, ok_b = (pa > 0.5).astype(int) == y, (pb > 0.5).astype(int) == y
    b_img, c_img = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p_img = float(binomtest(b_img, b_img + c_img, 0.5).pvalue) if b_img + c_img else 1.0
    uniq, idx = group_index(groups)
    d_k = np.array([int((~ok_a[i] & ok_b[i]).sum()) - int((ok_a[i] & ~ok_b[i]).sum()) for i in idx])
    n_k = np.array([len(i) for i in idx])
    x2 = d_k.sum() ** 2 / (d_k ** 2).sum() if (d_k ** 2).sum() else 0.0
    p_clu = float(chi2.sf(x2, 1))
    diff = (ok_b.mean() - ok_a.mean())
    boots = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots[r] = d_k[pick].sum() / n_k[pick].sum()
    p_boot = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"only_ref_correct": b_img, "only_compared_correct": c_img, "p_mcnemar_images": p_img,
            "acc_diff": diff, "acc_diff_ci_lo": float(np.percentile(boots, 2.5)),
            "acc_diff_ci_hi": float(np.percentile(boots, 97.5)), "p_bootstrap_groups": p_boot,
            "x2_durkalski": float(x2), "p_mcnemar_clustered": p_clu,
            "groups_with_discordance": int((d_k != 0).sum()), "n_groups": int(len(uniq))}


def boot_ci(y, p, groups, key, rng, n_boot=2000):
    uniq, idx = group_index(groups)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(y[pick])) < 2:
            continue
        vals.append(metrics(y[pick], p[pick])[key])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wide(pred):
    w = pred.pivot(index="path", columns="model", values="p_spoiled")
    meta = pred.drop_duplicates("path").set_index("path").loc[w.index, ["session", "label"]]
    return w, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default="D:/Research_Work/paperpwork/AI/MeatGuard_II-paper")
    ap.add_argument("--match-inliers", type=int, default=30)
    args = ap.parse_args()
    root = Path(args.paper_dir)
    out = root / "runs_specimen" / "reviewer_analyses"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Cluster-aware paired comparisons, specimen-level split
    w, meta = wide(pd.read_csv(root / "runs_specimen/results/predictions.csv"))
    y, g = meta["label"].to_numpy(), meta["session"].to_numpy()
    pairs = [("meatguard_s:distilled", m) for m in ["teacher:resnet50", "baseline:efficientnet_b0",
             "baseline:mobilenet_v3_small", "baseline:shufflenet_v2_x1_0", "meatguard_s:scratch", "meatguard_s:pruned"]]
    pairs += [("meatguard_s:pruned", "meatguard_s:int8"), ("meatguard_s:pruned", "meatguard_s:deployed"),
              ("meatguard_s:pruned", "baseline:mobilenet_v3_small"), ("meatguard_s:deployed", "baseline:mobilenet_v3_small"),
              ("baseline:mobilenet_v3_small", "baseline:efficientnet_b0"), ("baseline:mobilenet_v3_small", "teacher:resnet50"),
              ("baseline:mobilenet_v3_small", "baseline:shufflenet_v2_x1_0")]
    rows = [{"reference": a, "compared": b, **paired_comparison(y, w[a].to_numpy(), w[b].to_numpy(), g, rng)}
            for a, b in pairs]
    paired = pd.DataFrame(rows)
    paired.to_csv(out / "paired_tests_clustered.csv", index=False)

    # 2. Leakage subset analysis, capture-session split
    nb = pd.read_csv(root / "runs/specimen_check/neighbours.csv")
    nb = nb[(nb["query_split"] == "test") & (nb["pair"] == "test->train") & (nb["pool"] == "same_class") & (nb["rank"] < 3)]
    best = nb.assign(best=nb[["inliers", "inliers_flipped"]].max(axis=1)).groupby("query")["best"].max()
    ws, ms = wide(pd.read_csv(root / "runs/results/predictions.csv"))
    best = best.reindex(ws.index).fillna(0)
    matched = (best >= args.match_inliers).to_numpy()
    ys, gs = ms["label"].to_numpy(), ms["session"].to_numpy()
    sub_rows = []
    for model in ws.columns:
        for name, mask in [("all", np.ones_like(matched)), ("matched", matched), ("unmatched", ~matched)]:
            p = ws[model].to_numpy()[mask]
            r = {"model": model, "subset": name, "n_sessions": int(len(np.unique(gs[mask]))), **metrics(ys[mask], p)}
            lo, hi = boot_ci(ys[mask], p, gs[mask], "macro_f1", rng)
            r.update({"macro_f1_ci_lo": lo, "macro_f1_ci_hi": hi})
            blo, bhi = boot_ci(ys[mask], p, gs[mask], "balanced_accuracy", rng)
            r.update({"balanced_accuracy_ci_lo": blo, "balanced_accuracy_ci_hi": bhi})
            sub_rows.append(r)
    subsets = pd.DataFrame(sub_rows)
    subsets.to_csv(out / "leakage_subsets_session_split.csv", index=False)

    # Specimen-split reference values for the same models, for direct comparison
    spec = pd.DataFrame([{"model": m, **metrics(y, w[m].to_numpy())} for m in w.columns])
    spec.to_csv(out / "specimen_split_reference.csv", index=False)

    summary = {"n_boot_paired": N_BOOT, "seed": SEED, "match_inliers": args.match_inliers,
               "session_split_test_matched": int(matched.sum()), "session_split_test_unmatched": int((~matched).sum()),
               "matched_by_class": {"fresh": int(matched[ys == 0].sum()), "spoiled": int(matched[ys == 1].sum())},
               "unmatched_by_class": {"fresh": int((~matched)[ys == 0].sum()), "spoiled": int((~matched)[ys == 1].sum())}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(json.dumps(summary, indent=2))
    print(paired.round(4).to_string())
    print(subsets.round(3).to_string())
    print(spec.round(3).to_string())


if __name__ == "__main__":
    main()
