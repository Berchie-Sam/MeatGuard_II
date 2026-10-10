"""Classification metrics shared by teacher and student training (binary fresh/spoiled)."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score


@torch.no_grad()
def predict(model, loader, device):
    """Return (labels, probabilities[:, C]) over a loader, with the model in eval mode."""
    model.eval()
    ys, ps = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), dim=1).cpu())
        ys.append(labels)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def classification_metrics(y_true, probs, positive=1):
    """Accuracy, macro-F1, recall of the positive class (spoiled = 1), ROC-AUC and the confusion matrix."""
    y_pred = probs.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=positive)),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(probs.shape[1]))).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probs[:, positive]))
    except ValueError:                     # only one class present
        out["roc_auc"] = float("nan")
    return out


def bootstrap_ci(y_true, probs, groups, metric="macro_f1", n_boot=2000, seed=0, positive=1):
    """95% CI of a metric by resampling *groups* (capture sessions), not individual images."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_of[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y_true[pick])) < 2:
            continue
        vals.append(classification_metrics(y_true[pick], probs[pick], positive)[metric])
    if not vals:                           # e.g. only one class present in the evaluated set
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar_test(y_true, pred_a, pred_b):
    """Exact (binomial) McNemar test on paired predictions; b = A right/B wrong, c = A wrong/B right."""
    from scipy.stats import binomtest
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}
