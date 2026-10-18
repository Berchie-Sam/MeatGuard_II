"""
Check whether the same piece of meat appears in more than one partition of the session split.

Sessions link near-identical frames by a perceptual hash (src/data/sessions.py), which cannot recognise one piece of
meat photographed at another zoom or angle, or mirrored. This check uses instance-level features instead:

1. Retrieval: DINOv2 ViT-S/14 embeddings of every crop and of its mirror image. Each val/test image (either
   orientation) is compared with every train image (and each test image with every val image); the nearest
   neighbours by cosine similarity are kept.
2. Verification: AKAZE keypoints (image upscaled x2) matched between the query and each of its nearest neighbours,
   in both orientations of the query, with RANSAC (similarity transform); the number of inliers is recorded.
3. Calibration: the same statistics against an equally sized pool of other-class images, as in the leakage report
   of sessions.py, and a positive control (pairs known to show one piece in different sessions).

Writes to --out (default runs/specimen_check/): neighbours.csv (per query image: its nearest neighbours with cosine,
orientation and inliers), session_pairs.csv (per query session: the closest session of the other partition),
summary.json (counts at several thresholds) and contact sheets of the closest cross-partition pairs for inspection.

Usage:
    python -m src.data.specimen_check --data data/processed/full/crops --split-file data/processed/full/sessions.csv
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CONTROL = [("Spoiled_CowMeat/Spoiled (4258).jpg", "Spoiled_CowMeat/Spoiled (4213).jpg"),
           ("Spoiled_CowMeat/Spoiled (4213).jpg", "Spoiled_CowMeat/Spoiled (4216).jpg"),
           ("Spoiled_CowMeat/Spoiled (4258).jpg", "Spoiled_CowMeat/Spoiled (4216).jpg")]


def load_dinov2():
    """DINOv2 ViT-S/14 returning the class-token embedding: torch.hub (branch given explicitly, which skips the
    GitHub branch lookup), else the same weights through timm from the Hugging Face hub."""
    try:
        import torch
        return torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14", skip_validation=True, trust_repo=True)
    except Exception as e:
        print(f"torch.hub failed ({type(e).__name__}: {e}); loading DINOv2 ViT-S/14 through timm")
        import timm
        return timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)


def embed(paths, data_dir, model, device, batch=64):
    """L2-normalised embeddings of each image and of its mirror image: (N, D), (N, D)."""
    import torch
    torch.set_grad_enabled(False)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out, out_flip = [], []
    for s in range(0, len(paths), batch):
        imgs = [np.asarray(Image.open(data_dir / p).convert("RGB").resize((224, 224), Image.BICUBIC))
                for p in paths[s:s + batch]]
        x = torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).float().div(255).sub(mean).div(std)
        for t, store in ((x, out), (torch.flip(x, dims=[3]), out_flip)):
            store.append(torch.nn.functional.normalize(model(t).float(), dim=1).cpu())
        if (s // batch) % 20 == 0:
            print(f"  embedded {min(s + batch, len(paths))}/{len(paths)}")
    return torch.cat(out).numpy(), torch.cat(out_flip).numpy()


class Verifier:
    """AKAZE + RANSAC inlier count between two crops, trying the query in both orientations."""

    def __init__(self, data_dir, scale=2.0, max_cache=4000):
        self.data_dir, self.scale, self.cache, self.max_cache = data_dir, scale, {}, max_cache
        self.akaze = cv2.AKAZE_create(threshold=0.0005)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, path, flip=False):
        key = (path, flip)
        if key not in self.cache:
            g = cv2.imread(str(self.data_dir / path), cv2.IMREAD_GRAYSCALE)
            if flip:
                g = cv2.flip(g, 1)
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            self.cache[key] = self.akaze.detectAndCompute(g, None)
            if len(self.cache) > self.max_cache:                   # keep memory bounded
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def inliers(self, a, b, flip=False):
        (k1, d1), (k2, d2) = self.features(a, flip), self.features(b)
        if d1 is None or d2 is None or len(d1) < 6 or len(d2) < 6:
            return 0
        good = [m for m, n in (p for p in self.bf.knnMatch(d1, d2, k=2) if len(p) == 2)
                if m.distance < 0.8 * n.distance]
        if len(good) < 6:
            return 0
        p1 = np.float32([k1[m.queryIdx].pt for m in good])
        p2 = np.float32([k2[m.trainIdx].pt for m in good])
        _, mask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=8.0)
        return int(mask.sum()) if mask is not None else 0

    def best(self, a, b):
        """(inliers, flipped) for the better orientation of a against b."""
        return max((self.inliers(a, b, f), f) for f in (False, True))


def neighbours(table, emb, emb_flip, queries, pool, k):
    """For each query index: the k pool images with the highest cosine (best of both query orientations)."""
    q, qf, p = emb[queries], emb_flip[queries], emb[pool]
    rows = []
    for s in range(0, len(queries), 512):
        sim, sim_f = q[s:s + 512] @ p.T, qf[s:s + 512] @ p.T
        best = np.maximum(sim, sim_f)
        top = np.argsort(-best, axis=1)[:, :k]
        for r, qi in enumerate(queries[s:s + 512]):
            for rank, j in enumerate(top[r]):
                rows.append({"query": table.at[qi, "path"], "query_split": table.at[qi, "split"],
                             "query_session": table.at[qi, "session"], "query_class": table.at[qi, "class"],
                             "rank": rank, "match": table.at[pool[j], "path"],
                             "match_split": table.at[pool[j], "split"],
                             "match_session": table.at[pool[j], "session"],
                             "match_class": table.at[pool[j], "class"], "cosine": float(best[r, j]),
                             "flipped": bool(sim_f[r, j] > sim[r, j])})
    return pd.DataFrame(rows)


def contact_sheets(pairs, data_dir, out, per_sheet=40, cols=4, size=150):
    """Query | match pairs, one per query session, ordered by cosine."""
    for n, s in enumerate(range(0, len(pairs), per_sheet)):
        chunk = pairs.iloc[s:s + per_sheet]
        rows = -(-len(chunk) // cols)
        sheet = Image.new("RGB", (cols * (2 * size + 20), rows * (size + 28)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk.itertuples()):
            x, y = (i % cols) * (2 * size + 20), (i // cols) * (size + 28)
            q = Image.open(data_dir / r.query).convert("RGB").resize((size, size))
            if r.flipped:
                q = ImageOps.mirror(q)
            sheet.paste(q, (x, y))
            sheet.paste(Image.open(data_dir / r.match).convert("RGB").resize((size, size)), (x + size + 2, y))
            draw.text((x, y + size + 1), f"{r.query_split} s{r.query_session} {Path(r.query).stem[-10:]}"
                      f"{' (mirrored)' if r.flipped else ''}", fill="black")
            draw.text((x, y + size + 13), f"-> {r.match_split} s{r.match_session}  cos {r.cosine:.3f}  "
                      f"inl {r.inliers}", fill="black")
        path = out / f"sheet_{n + 1:02d}.png"
        sheet.save(path)
        print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="crops/ ImageFolder root")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out", default="runs/specimen_check")
    ap.add_argument("--k", type=int, default=3, help="neighbours verified per query image")
    ap.add_argument("--sheet-pairs", type=int, default=120, help="query sessions shown in the contact sheets")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    args = ap.parse_args()
    import torch
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.split_file).reset_index(drop=True)
    print(f"{len(table)} images, {table['session'].nunique()} sessions; device {args.device}")

    model = load_dinov2().to(args.device).eval()
    emb, emb_flip = embed(table["path"].tolist(), data_dir, model, args.device)
    np.savez_compressed(out / "embeddings.npz", emb=emb.astype(np.float16), emb_flip=emb_flip.astype(np.float16))

    idx = {s: table.index[table["split"] == s].to_numpy() for s in ("train", "val", "test")}
    rng = np.random.default_rng(0)
    verifier = Verifier(data_dir)
    frames, summary = [], {}
    for qs, ps in (("test", "train"), ("val", "train"), ("test", "val")):
        for cls in sorted(table["class"].unique()):
            queries = idx[qs][table.loc[idx[qs], "class"].to_numpy() == cls]
            same = idx[ps][table.loc[idx[ps], "class"].to_numpy() == cls]
            other = idx[ps][table.loc[idx[ps], "class"].to_numpy() != cls]
            other = rng.choice(other, min(len(same), len(other)), replace=False)   # equal-sized chance pool
            for pool_name, pool in (("same_class", same), ("other_class", other)):
                nb = neighbours(table, emb, emb_flip, queries, pool, args.k)
                nb["pair"], nb["pool"] = f"{qs}->{ps}", pool_name
                frames.append(nb)
        print(f"retrieval {qs} -> {ps} done")
    nb = pd.concat(frames, ignore_index=True)

    print(f"verifying {len(nb)} candidate pairs ...")
    res = [verifier.best(r.query, r.match) for r in nb.itertuples()]
    nb["inliers"], nb["inliers_flipped"] = [r[0] for r in res], [r[1] for r in res]
    nb.to_csv(out / "neighbours.csv", index=False)

    # Per query image: its best candidate (highest inliers, then cosine)
    best = (nb.sort_values(["inliers", "cosine"], ascending=False)
            .drop_duplicates(["pair", "pool", "query"]))
    for (pair, pool), g in best.groupby(["pair", "pool"]):
        summary[f"{pair}|{pool}"] = {
            "n_query_images": int(len(g)), "n_query_sessions": int(g["query_session"].nunique()),
            **{f"images_inliers>={t}": int((g["inliers"] >= t).sum()) for t in (15, 20, 30, 50)},
            **{f"sessions_inliers>={t}": int(g.loc[g["inliers"] >= t, "query_session"].nunique())
               for t in (15, 20, 30, 50)},
            **{f"images_cosine>={t}": int((g["cosine"] >= t).sum()) for t in (0.7, 0.8, 0.9)},
        }

    control = []
    for a, b in CONTROL:
        ia, ib = table.index[table["path"] == a], table.index[table["path"] == b]
        if len(ia) and len(ib):
            cos = float(max(emb[ia[0]] @ emb[ib[0]], emb_flip[ia[0]] @ emb[ib[0]]))
            inl, flipped = verifier.best(a, b)
            control.append({"a": a, "b": b, "cosine": cos, "inliers": inl, "flipped": flipped})
    summary["positive_control"] = control
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Session level, same-class pools only: the closest session of the other partition for each query session
    sess = (best[best["pool"] == "same_class"].sort_values(["inliers", "cosine"], ascending=False)
            .drop_duplicates(["pair", "query_session"]))
    sess.to_csv(out / "session_pairs.csv", index=False)
    contact_sheets(sess[sess["pair"] == "test->train"].head(args.sheet_pairs), data_dir, out)


if __name__ == "__main__":
    main()
