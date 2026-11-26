"""Re-evaluate a frozen proposal checkpoint without training or threshold selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import save_json, summarize
from .models import build_model, classify


def evaluate(experiment, kind, split, device="cpu"):
    result = json.loads((experiment / "results.json").read_text())
    audit = json.loads((experiment / "dataset_audit.json").read_text())
    checkpoint = torch.load(experiment / f"{kind}.pt", weights_only=True, map_location="cpu")
    if checkpoint["manifest_sha256"] != audit["manifest_sha256"]:
        raise ValueError("Checkpoint and dataset manifests differ.")
    selection = json.loads((experiment / "selection.json").read_text())["models"][kind]
    if selection != {"epoch": checkpoint["epoch"], "threshold": checkpoint["threshold"]}:
        raise ValueError("Checkpoint selection changed.")
    with np.load(experiment / "predictions.npz", allow_pickle=False) as saved:
        all_ids = sum((saved[f"{s}_ids"].tolist() for s in ("train", "valid", "test")), [])
        n_train = len(saved["train_ids"])
        start = n_train if split == "valid" else n_train + len(saved["valid_ids"])
        ids = saved[f"{split}_ids"]
        labels = saved[f"{split}_labels"]
        concurrent = saved[f"{split}_concurrent_labels"]
        prior = saved[f"{kind}_{split}_scores"]
    with np.load(experiment / "features.npz", allow_pickle=False) as cache:
        context = json.loads(cache["key"].item())
        if context["video_ids"] != all_ids or context["manifest_sha256"] != audit["manifest_sha256"]:
            raise ValueError("Feature cache provenance mismatch.")
        features = cache["features"][start:start + len(ids)]
    model = build_model(kind).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        scores = torch.cat([
            classify(model, batch.to(device)).sigmoid().cpu()
            for batch in torch.from_numpy(features).split(64)
        ]).numpy()
    return {
        "evaluation_unit": "whole_video", "split": split, "videos": len(ids),
        "model": kind, "checkpoint_epoch": checkpoint["epoch"],
        "fixed_threshold": checkpoint["threshold"],
        "max_score_difference_from_saved": float(np.max(np.abs(scores - prior))),
        "metrics": summarize(labels, scores, concurrent, audit["main_class_ids"],
                             checkpoint["classes"], checkpoint["threshold"]),
        "saved_metrics": result["models"][kind]["validation" if split == "valid" else "test"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("proposal/results"))
    parser.add_argument("--model", choices=("mean", "bigru_mean", "bigru_attention"), required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    result = evaluate(args.experiment, args.model, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("metrics", "saved_metrics")}))


if __name__ == "__main__":
    main()
