"""Independent clip-level 31-class evaluation; no prediction I/O or Test access."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

RELEASE = Path(__file__).resolve().parents[1] / "release_v1"


def _mapping():
    names = [r["name"] for r in json.loads((RELEASE / "class_mapping.json").read_text(encoding="utf-8"))["classes"]]
    ids = json.loads((RELEASE / "main_evaluation_classes.json").read_text(encoding="utf-8"))["class_ids"]
    return names, ids


def average_precision(target, scores):
    """Noninterpolated AP, grouping tied scores; no positives -> None."""
    y = np.asarray(target, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or len(y) == 0 or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("AP requires aligned nonempty finite scores and binary targets")
    positives = y.sum()
    if positives == 0:
        return None
    order = np.argsort(-s, kind="stable")
    ordered_y, ordered_s = y[order], s[order]
    ends = np.r_[np.flatnonzero(ordered_s[:-1] != ordered_s[1:]), len(y) - 1]
    cumulative = np.cumsum(ordered_y)[ends]
    positive_increments = np.diff(np.r_[0, cumulative])
    return float(np.dot(positive_increments, cumulative / (ends + 1)) / positives)


def _inputs(logits, target):
    z = torch.as_tensor(logits, dtype=torch.float64, device="cpu").detach()
    y = torch.as_tensor(target, dtype=torch.float64, device="cpu").detach()
    if (z.ndim != 2 or z.shape != y.shape or z.shape[0] < 1 or z.shape[1] != 119
            or not torch.isfinite(z).all() or not torch.isfinite(y).all() or not torch.all((y == 0) | (y == 1))):
        raise ValueError("Expected aligned finite logits and binary labels [N,119]")
    return z, y


def _confusion(y, probabilities, threshold):
    pred, truth = probabilities >= threshold, y.astype(bool)
    tp = np.sum(pred & truth, axis=0)
    fp = np.sum(pred & ~truth, axis=0)
    fn = np.sum(~pred & truth, axis=0)
    tn = np.sum(~pred & ~truth, axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=tp + fn > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=2 * tp + fp + fn > 0)
    return tp, fp, tn, fn, precision, recall, f1


def evaluate_multilabel(logits, target, *, threshold=0.5, require_all_main_classes=True):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid independent sigmoid threshold")
    z, labels = _inputs(logits, target)
    probabilities, y = z.sigmoid().numpy(), labels.numpy()
    names, ids = _mapping()
    aps = [average_precision(y[:, i], probabilities[:, i]) for i in range(119)]
    missing = [i for i in ids if aps[i] is None]
    if require_all_main_classes and missing:
        raise ValueError("Missing main-class positives: never call a small functional batch full 31-class mAP")
    tp, fp, tn, fn, precision, recall, f1 = _confusion(y, probabilities, threshold)
    return dict(evaluation_unit="action_interval_clip", num_samples=len(y), main_class_ids=ids,
                main_mAP=float(np.mean([aps[i] for i in ids])) if not missing else None,
                main_macro_f1=float(f1[ids].mean()), BCE119=float(F.binary_cross_entropy_with_logits(z, labels)),
                threshold=float(threshold), missing_main_class_ids=missing,
                per_class=[dict(class_id=i, name=names[i], main_evaluation=i in ids, positive_clips=int(y[:, i].sum()), AP=aps[i],
                                TP=int(tp[i]), FP=int(fp[i]), TN=int(tn[i]), FN=int(fn[i]),
                                Precision=float(precision[i]), Recall=float(recall[i]), F1=float(f1[i])) for i in range(119)])


def select_validation_threshold(logits, target, *, split):
    if split != "valid":
        raise ValueError("Threshold selection is allowed on Valid only")
    z, labels = _inputs(logits, target)
    _, ids = _mapping()
    y, probabilities = labels.numpy(), z.sigmoid().numpy()
    if any(y[:, i].sum() == 0 for i in ids):
        raise ValueError("Threshold selection requires all fixed 31 main classes")
    grid = [i / 100 for i in range(5, 96)]
    values = [float(_confusion(y, probabilities, t)[-1][ids].mean()) for t in grid]
    # Earliest ascending-grid maximum: deterministic lower-threshold tie break.
    j = int(np.argmax(values))
    return dict(threshold=grid[j], valid_macro_f1=values[j], criterion="fixed31 Macro-F1 on Valid",
                candidates=grid, macro_f1_curve=values, tie_rule="lowest threshold", split="valid")
