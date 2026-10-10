"""
Faithfulness measures for attribution maps (Petsiuk et al., 2018; Adebayo et al., 2018).

All functions take a normalised input tensor of shape (1, C, H, W). "Removing" a pixel sets it to 0 in
normalised space, i.e. the ImageNet mean colour, which is also the background fill of the cropped images.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def _to_hw(attribution, H, W):
    """Attribution map as a float array of shape (H, W); smaller maps are upsampled bilinearly."""
    a = torch.as_tensor(np.asarray(attribution, dtype=np.float32))
    if a.shape != (H, W):
        a = F.interpolate(a[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return a.numpy()


@torch.no_grad()
def perturbation_curve(model, image, attribution, target_class, mode="deletion", steps=10, order=None,
                       batch_size=11):
    """
    Class probability as pixels are changed in decreasing order of attribution.

    deletion: start from the image and replace the top-ranked pixels with the baseline (mean colour).
    insertion: start from the baseline and restore the top-ranked pixels.
    Returns (fractions t = 0, 1/steps, ..., 1; probabilities f(t)).
    """
    model.eval()
    _, C, H, W = image.shape
    if order is None:
        order = np.argsort(-_to_hw(attribution, H, W).ravel(), kind="stable")
    rank = np.empty(H * W, dtype=np.int64)
    rank[order] = np.arange(H * W)
    rank = torch.as_tensor(rank.reshape(H, W), device=image.device)
    fractions = np.linspace(0.0, 1.0, steps + 1)
    baseline = torch.zeros_like(image)
    variants = []
    for t in fractions:
        changed = (rank < int(round(t * H * W)))[None, None]          # top-ranked pixels
        if mode == "deletion":
            variants.append(torch.where(changed, baseline, image))
        else:
            variants.append(torch.where(changed, image, baseline))
    batch = torch.cat(variants)
    probs = torch.cat([torch.softmax(model(batch[i:i + batch_size]), dim=1)[:, target_class]
                       for i in range(0, len(batch), batch_size)])
    return fractions, probs.float().cpu().numpy()


def curve_auc(fractions, probs):
    """Normalised area under f(t) on t in [0, 1] (trapezoidal rule), Eq. (6)."""
    return float(np.trapz(probs, fractions))


def calculate_deletion_auc(model, image, mask, target_class, steps=10):
    """Deletion AUC (lower = more faithful). `mask` is the attribution map."""
    return curve_auc(*perturbation_curve(model, image, mask, target_class, "deletion", steps))


def calculate_insertion_auc(model, image, mask, target_class, steps=10):
    """Insertion AUC (higher = more faithful). `mask` is the attribution map."""
    return curve_auc(*perturbation_curve(model, image, mask, target_class, "insertion", steps))


def random_order_auc(model, image, target_class, mode="deletion", steps=10, repeats=5, seed=0):
    """Mean AUC for random pixel orderings: the reference a faithful map should beat."""
    rng = np.random.default_rng(seed)
    _, _, H, W = image.shape
    aucs = [curve_auc(*perturbation_curve(model, image, None, target_class, mode, steps,
                                          order=rng.permutation(H * W))) for _ in range(repeats)]
    return float(np.mean(aucs))


def meat_share(attribution, mask):
    """
    Share of positive attribution on the meat (Eq. 7) and the share expected by chance (mask area).
    `mask` is a boolean (H, W) meat mask in the same (cropped) frame as the model input.
    """
    mask = np.asarray(mask, dtype=bool)
    a = np.clip(_to_hw(attribution, *mask.shape), 0, None)
    total = a.sum()
    share = float(a[mask].sum() / total) if total > 0 else float("nan")
    return {"meat_share": share, "mask_area": float(mask.mean())}


def calculate_iou(mask_a, mask_b):
    """
    Intersection over Union between two binary masks.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / (union + 1e-7)


def randomization_sanity_check(model, explain, image, target_class, blocks=None, seed=0):
    """
    Cascading model-parameter randomisation (Adebayo et al., 2018).

    Re-initialises the model's top-level blocks one at a time, from the output towards the input, and
    records the Spearman correlation between the original map and the map of the randomised model.
    `explain(model, image, target_class)` must return an attribution map. Correlations that stay high
    indicate a map that does not depend on what the model has learned.
    """
    torch.manual_seed(seed)
    reference = np.asarray(explain(model, image, target_class), dtype=np.float32).ravel()
    randomised = copy.deepcopy(model).eval()
    names = blocks or [n for n, m in reversed(list(randomised.named_children()))
                       if any(True for _ in m.parameters())]
    results = []
    for name in names:
        for module in getattr(randomised, name).modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        rho = spearmanr(reference, np.asarray(explain(randomised, image, target_class), dtype=np.float32).ravel())
        results.append({"randomised_up_to": name, "spearman": float(rho.correlation)})
    return results
