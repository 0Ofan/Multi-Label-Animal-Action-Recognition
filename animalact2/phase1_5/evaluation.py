"""Reuse fixed31 metrics; add Train support flags, parent counts and result exports."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from animalact2.phase1_5_1.evaluation import evaluate_multilabel, select_validation_threshold, average_precision
from .safety import OUT, write_json


def evaluate_outputs(logits, target, parent_video_ids, policy, *, threshold=.5, require_all_main_classes=True):
    result = evaluate_multilabel(logits, target, threshold=threshold, require_all_main_classes=require_all_main_classes)
    z = torch.as_tensor(logits).detach().cpu().float()
    y = torch.as_tensor(target).detach().cpu().float()
    if len(parent_video_ids) != len(y):
        raise ValueError("Predictions and parent identities are misaligned")
    mask = policy.active_mask.cpu()
    result["BCE_Train_supported118"] = float(F.binary_cross_entropy_with_logits(z[:, mask], y[:, mask]))
    result["Train_class_policy"] = policy.metadata()
    for row in result["per_class"]:
        i = row["class_id"]
        row["positive_parent_videos"] = len({p for p, label in zip(parent_video_ids, y[:, i].tolist()) if label == 1})
        row["Train_positive_clips"] = int(policy.train_positive_counts[i])
        row["learning_status"] = "Train_supported" if bool(mask[i]) else "unsupported_from_Train_not_normally_learnable"
        row["eligible_for_prediction"] = bool(mask[i])
    return result


def action_predictions(logits, threshold, policy):
    # Preserve 119 raw logits; unsupported outputs cannot be emitted as learned actions.
    return (logits.sigmoid() >= threshold) & policy.active_mask.to(logits.device)[None, :]


def save_evaluation(directory, result):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUT):
        raise ValueError("Evaluation output outside independent phase")
    write_json(directory / "metrics.json", result)
    rows = result["per_class"]
    for name, subset in (("per_class_119.csv", rows), ("per_class_main31.csv", [r for r in rows if r["main_evaluation"]])):
        with (directory / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    matrix = [dict(class_id=r["class_id"], name=r["name"], TP=r["TP"], FP=r["FP"], TN=r["TN"], FN=r["FN"],
                   Precision=r["Precision"], Recall=r["Recall"], threshold=result["threshold"],
                   one_vs_rest_2x2=json.dumps([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])) for r in rows if r["main_evaluation"]]
    with (directory / "confusion_main31.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
