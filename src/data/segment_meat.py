"""
Offline meat segmentation for MeatScan (background-shortcut mitigation).

Pipeline per image:
    Grounding DINO (text prompt "raw meat.") -> box prompts -> SAM 2.1 -> binary mask

Outputs under --out (ImageFolder-compatible class sub-folders):
    masks/<class>/<name>.png    binary meat mask (255 = meat)
    masked/<class>/<name>.jpg   full frame, background filled with the ImageNet mean colour
    crops/<class>/<name>.jpg    masked image cropped to the padded mask box, resized to 224
    segmentation.csv            per-image record (scores, boxes, mask area, fallback flags);
                                fallback: 0 = box-prompted SAM mask, 1 = nothing detected (whole-frame prompt),
                                2 = box mask < 2% of the frame, redone with point prompts (or the boxes themselves)

The run is resumable: images whose mask already exists are skipped.

Usage:
    python -m src.data.segment_meat --src E:/Datasets/MeatScan --out E:/Datasets/full
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_SIZE = 224
BG_FILL = (124, 116, 104)  # ImageNet mean in RGB -> ~0 after normalisation
PROMPT = "raw meat."
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_FIELDS = ["path", "class", "n_boxes", "best_score", "boxes", "mask_area", "fallback", "small_mask"]
SMALL_MASK = 0.02   # masks below this share of the frame are retried (fallback = 2 in segmentation.csv)


def find_images(src):
    return sorted(
        p for p in Path(src).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.parent != Path(src)
    )


def select_boxes(boxes, scores, min_score, rel_score):
    """Keep every confident box (multiple cuts in one frame are common)."""
    if len(scores) == 0:
        return [], 0.0
    best = float(scores.max())
    keep = (scores >= min_score) & (scores >= rel_score * best)
    if not keep.any():
        keep = scores == scores.max()
    return boxes[keep].tolist(), best


def sam_mask(sam, rgb, imgsz, **prompts):
    """Union of the SAM masks for the given prompts, at the image's own resolution."""
    h, w = rgb.shape[:2]
    res = sam(rgb[:, :, ::-1], imgsz=imgsz, verbose=False, **prompts)[0]
    if res.masks is None:
        return np.zeros((h, w), bool)
    mask = res.masks.data.cpu().numpy().astype(bool).any(0)
    if mask.shape != (h, w):
        mask = np.asarray(Image.fromarray(mask).resize((w, h), Image.NEAREST))
    return mask


def crop_window(mask, pad=0.08):
    """Square crop window (x0, y0, side) around the mask, enlarged by `pad` per side and kept in frame."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = min(int(max(y1 - y0, x1 - x0) * (1 + 2 * pad)), max(h, w))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    # Shift the square window back inside the frame so no artificial border is added
    x0 = int(np.clip(cx - side // 2, 0, max(w - side, 0)))
    y0 = int(np.clip(cy - side // 2, 0, max(h - side, 0)))
    return x0, y0, side


def crop_to_mask(masked, mask, pad=0.08):
    x0, y0, side = crop_window(mask, pad)
    # Square crop on a background-filled canvas (only matters for non-square frames)
    canvas = Image.new("RGB", (side, side), BG_FILL)
    canvas.paste(Image.fromarray(masked), (-x0, -y0))
    return canvas.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def crop_mask(mask, pad=0.08):
    """The meat mask in the frame of the cropped image (same window and size as `crop_to_mask`)."""
    x0, y0, side = crop_window(mask, pad)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(Image.fromarray(mask.astype(np.uint8) * 255), (-x0, -y0))
    return np.asarray(canvas.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127


def load_cropped_mask(mask_path, pad=0.08):
    """Read a saved full-frame mask (masks/<class>/<name>.png) and return it in the cropped frame."""
    return crop_mask(np.asarray(Image.open(mask_path)) > 127, pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="artifacts/weights")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--det-edge", type=int, default=448)
    ap.add_argument("--sam-imgsz", type=int, default=512)
    ap.add_argument("--min-score", type=float, default=0.30)
    ap.add_argument("--rel-score", type=float, default=0.60)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (smoke test)")
    args = ap.parse_args()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from ultralytics import SAM

    torch.set_num_threads(args.threads)
    weights = Path(args.weights)
    hf_cache = str(weights / "hf")
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", cache_dir=hf_cache)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny", cache_dir=hf_cache).eval()
    sam = SAM(str(weights / "sam2.1_t.pt"))

    src, out = Path(args.src), Path(args.out)
    images = find_images(src)
    todo = [p for p in images
            if not (out / "masks" / p.relative_to(src)).with_suffix(".png").exists()]
    print(f"{len(images)} images, {len(images) - len(todo)} already done, {len(todo)} to process", flush=True)
    if args.limit:
        todo = todo[:args.limit]

    csv_path = out / "segmentation.csv"
    out.mkdir(parents=True, exist_ok=True)
    new_csv = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_csv:
            writer.writeheader()

        t0 = time.time()
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            pils = [Image.open(p).convert("RGB") for p in batch]
            inp = proc(images=pils, text=[PROMPT] * len(pils), return_tensors="pt",
                       size={"shortest_edge": args.det_edge, "longest_edge": 1333})
            with torch.inference_mode():
                det_out = det(**inp)
            results = proc.post_process_grounded_object_detection(
                det_out, inp.input_ids, box_threshold=0.25, text_threshold=0.25,
                target_sizes=[im.size[::-1] for im in pils])

            for path, im, res in zip(batch, pils, results):
                rgb = np.asarray(im)
                h, w = rgb.shape[:2]
                boxes, best = select_boxes(res["boxes"], res["scores"], args.min_score, args.rel_score)
                fallback = not boxes
                if fallback:  # nothing detected: let SAM pick the dominant object in the frame
                    boxes = [[0, 0, w, h]]
                boxes = [[float(np.clip(b[0], 0, w)), float(np.clip(b[1], 0, h)),
                          float(np.clip(b[2], 0, w)), float(np.clip(b[3], 0, h))] for b in boxes]
                mask = sam_mask(sam, rgb, args.sam_imgsz, bboxes=boxes)
                if not mask.any():
                    mask = np.ones((h, w), bool)
                    fallback = True
                elif not fallback and mask.mean() < SMALL_MASK:
                    # Box prompt latched onto something else (e.g. black padding of a rotated copy):
                    # retry with a point at each box centre, then fall back to the boxes themselves.
                    centres = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
                    mask = sam_mask(sam, rgb, args.sam_imgsz, points=centres, labels=[1] * len(centres))
                    if mask.mean() < SMALL_MASK:
                        mask = np.zeros((h, w), bool)
                        for x0, y0, x1, y1 in boxes:
                            mask[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
                    fallback = 2

                masked = rgb.copy()
                masked[~mask] = BG_FILL
                rel = path.relative_to(src)
                for sub, img, ext in (("masks", Image.fromarray(mask.astype(np.uint8) * 255), ".png"),
                                      ("masked", Image.fromarray(masked), ".jpg"),
                                      ("crops", crop_to_mask(masked, mask), ".jpg")):
                    dst = (out / sub / rel).with_suffix(ext)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img.save(dst, quality=95) if ext == ".jpg" else img.save(dst)

                area = float(mask.mean())
                writer.writerow({
                    "path": rel.as_posix(), "class": rel.parts[0], "n_boxes": 0 if fallback == 1 else len(boxes),
                    "best_score": round(best, 4), "boxes": ";".join(",".join(f"{v:.1f}" for v in b) for b in boxes),
                    "mask_area": round(area, 4), "fallback": int(fallback), "small_mask": int(area < SMALL_MASK),
                })
            fh.flush()

            done = start + len(batch)
            rate = (time.time() - t0) / done
            print(f"{done}/{len(todo)}  {rate:.2f}s/img  eta {(len(todo) - done) * rate / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
