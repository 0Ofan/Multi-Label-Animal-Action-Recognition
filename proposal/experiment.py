"""Train the three proposed clip-level classifiers on the fixed AnimalAct splits."""

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

from .data import load_dataset
from .models import build_model, classify


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def average_precision(labels, scores):
    import numpy as np

    order = np.argsort(-scores, kind="stable")
    positives = labels[order].cumsum()
    if positives[-1] == 0:
        return None
    # Equal scores share one precision-recall operating point.
    ranks = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
    detected = positives[ranks]
    return float((np.diff(np.r_[0, detected]) * detected / (ranks + 1)).sum() / positives[-1])


def summarize(labels, scores, concurrent, class_ids, classes, threshold):
    import numpy as np

    y, score = labels[:, class_ids], scores[:, class_ids]
    predicted = score >= threshold
    tp = (y & predicted).sum(axis=0)
    fp = (~y & predicted).sum(axis=0)
    fn = (y & ~predicted).sum(axis=0)
    tn = (~y & ~predicted).sum(axis=0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    ap = [average_precision(y[:, k], score[:, k]) for k in range(len(class_ids))]
    supported = [v for v in ap if v is not None]
    c = concurrent[:, class_ids]
    subset = c.any(axis=1)
    subset_predictions = int(predicted[subset].sum())
    subset_correct = int((y[subset] & predicted[subset]).sum())
    concurrent_total = int(c.sum())
    concurrent_detected = int((c & predicted).sum())
    return {
        "map": float(np.mean(supported)) if supported else None,
        "macro_f1": float(f1.mean()),
        "classes_with_positive_support": len(supported),
        "concurrent_subset": {
            "videos": int(subset.sum()),
            "concurrent_labels": concurrent_total,
            "detected_concurrent_labels": concurrent_detected,
            "recall": concurrent_detected / concurrent_total if concurrent_total else None,
            "primary_predictions": subset_predictions,
            "correct_primary_predictions": subset_correct,
            "precision": subset_correct / subset_predictions if subset_predictions else None,
        },
        "per_class": [
            {
                "class_id": index, "action": classes[index], "ap": ap[k],
                "positive_videos": int(tp[k] + fn[k]), "f1": float(f1[k]),
                "confusion_2x2_tn_fp_fn_tp": [[int(tn[k]), int(fp[k])], [int(fn[k]), int(tp[k])]],
            }
            for k, index in enumerate(class_ids)
        ],
    }


def sample_frames(path, count):
    import cv2
    import numpy as np

    decoder = cv2.VideoCapture(str(path))
    try:
        total = int(decoder.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            raise ValueError("Video has no readable frame count")
        indices = ((np.arange(count) + 0.5) * total / count).astype(int)
        selected = {}
        targets = set(indices.tolist())
        for index in range(int(indices[-1]) + 1):
            if not decoder.grab():
                break
            if index in targets:
                ok, frame = decoder.retrieve()
                if ok:
                    selected[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if len(selected) != len(targets):
            raise ValueError(f"Decoded {len(selected)} of {len(targets)} requested frame positions")
        return [selected[index] for index in indices]
    finally:
        decoder.release()


def extract_features(archive, rows, args, metadata, device):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    settings = {
        "manifest_sha256": metadata["manifest_sha256"],
        "video_ids": [row["id"] for row in rows],
        "frames": args.frames,
        "encoder": "ResNet18_Weights.IMAGENET1K_V1",
        "torchvision": __import__("torchvision").__version__,
    }
    key = json.dumps(settings, sort_keys=True)
    path = args.output / "features.npz"
    expected = (len(rows), args.frames, 512)
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            if cache["key"].item() == key and cache["features"].shape == expected:
                print(f"Reusing features {expected}")
                return cache["features"]
        raise ValueError(f"Feature cache has different settings. Use a new output directory: {path.parent}")

    weights = ResNet18_Weights.IMAGENET1K_V1
    if args.weights:
        with args.weights.open("rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        if digest != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
            raise ValueError("Local file is not the official ResNet-18 weight file")
        encoder = resnet18()
        encoder.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    else:
        encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval().requires_grad_(False).to(device)
    preprocess = weights.transforms()
    features = []
    batch = []
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix="animalact-") as tmp, torch.inference_mode():
        clip = Path(tmp) / "clip.mp4"
        for index, row in enumerate(rows, 1):
            with archive.open(row["path"]) as source, clip.open("wb") as output:
                shutil.copyfileobj(source, output)
            try:
                batch.extend(preprocess(Image.fromarray(frame)) for frame in sample_frames(clip, args.frames))
            except ValueError as exc:
                raise ValueError(f"{row['id']}: {exc}") from exc
            if len(batch) >= 32 or index == len(rows):
                tensor = torch.stack(batch).to(device)
                features.append(encoder(tensor).cpu().numpy())
                batch.clear()
            if index % 100 == 0 or index == len(rows):
                print(f"Extracted {index}/{len(rows)} videos")
    matrix = np.concatenate(features).reshape(expected)
    assert np.isfinite(matrix).all()
    pending = path.with_suffix(".tmp.npz")
    np.savez_compressed(pending, features=matrix, key=key)
    pending.replace(path)
    return matrix


def train_models(features, rows, metadata, args, device):
    import numpy as np
    import torch
    from torch import nn

    classes, main = metadata["classes"], metadata["main_class_ids"]
    mode_names = ("mean", "bigru_mean", "bigru_attention")
    arrays = {}
    offset = 0
    for split, samples in rows.items():
        end = offset + len(samples)
        y = np.zeros((len(samples), len(classes)), dtype=bool)
        concurrent = np.zeros_like(y)
        for i, sample in enumerate(samples):
            y[i, sample["labels"]] = True
            concurrent[i, sample["concurrent"]] = True
        arrays[split] = (torch.from_numpy(features[offset:end]).to(device), y, concurrent)
        offset = end

    def predict(model, x):
        model.eval()
        with torch.inference_mode():
            return torch.cat([classify(model, batch).sigmoid().cpu() for batch in x.split(64)]).numpy()

    def validation_map(y, scores):
        ap = [average_precision(y[:, index], scores[:, index]) for index in main]
        if any(item is None for item in ap):
            raise ValueError("Validation data is missing a primary class")
        return float(np.mean(ap))

    train_x, train_y, _ = arrays["train"]
    valid_x, valid_y, valid_c = arrays["valid"]
    train_targets = torch.from_numpy(train_y.astype("float32")).to(device)
    hist = {}
    report = {}
    predictions = {}
    for name in mode_names:
        torch.manual_seed(args.seed)
        model = build_model(name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batches = torch.Generator().manual_seed(args.seed)
        criterion = nn.BCEWithLogitsLoss()
        best_map = -1
        hist[name] = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for indices in torch.randperm(len(train_x), generator=batches).split(32):
                indices = indices.to(device)
                optimizer.zero_grad()
                loss = criterion(classify(model, train_x[indices]), train_targets[indices])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total_loss += loss.item() * len(indices)
            valid_scores = predict(model, valid_x)
            ap = validation_map(valid_y, valid_scores)
            avg_loss = total_loss / len(train_x)
            if not np.isfinite(avg_loss) or not np.isfinite(valid_scores).all():
                raise ValueError(f"Non-finite loss or predictions for {name}")
            hist[name].append({"epoch": epoch, "train_bce": avg_loss, "valid_map": ap})
            if ap > best_map:
                best_map = ap
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"{name} epoch {epoch}: train BCE={avg_loss:.4f}, validation mAP={ap:.4f}")
            if epoch - best_epoch >= 8:
                break

        model.load_state_dict(best_state)
        valid_scores = predict(model, valid_x)
        grid = sorted(range(5, 96, 5), key=lambda value: abs(value - 50))
        threshold = max(
            (value / 100 for value in grid),
            key=lambda t: summarize(valid_y, valid_scores, valid_c, main, classes, t)["macro_f1"],
        )
        torch.save(
            {"model": name, "state_dict": best_state, "classes": classes, "frames": args.frames,
             "threshold": threshold, "epoch": best_epoch, "encoder": "ResNet18_Weights.IMAGENET1K_V1",
             "manifest_sha256": metadata["manifest_sha256"], "evaluation_unit": "whole_video"},
            args.output / f"{name}.pt",
        )
        report[name] = {
            "selected_epoch": best_epoch,
            "threshold": threshold,
            "validation": summarize(valid_y, valid_scores, valid_c, main, classes, threshold),
        }
        predictions[f"{name}_valid_scores"] = valid_scores
        save_json(args.output / "selection.json", {
            "selection_split": "valid", "main_class_ids": main,
            "models": {key: {"epoch": value["selected_epoch"], "threshold": value["threshold"]}
                       for key, value in report.items()},
        })
    # Freeze every model's selection before any test prediction is computed.
    if "test" in arrays:
        for name in mode_names:
            model = build_model(name).to(device)
            checkpoint = torch.load(args.output / f"{name}.pt", weights_only=True, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            test_x, test_y, test_c = arrays["test"]
            scores = predict(model, test_x)
            report[name]["test"] = summarize(test_y, scores, test_c, main, classes, checkpoint["threshold"])
            predictions[f"{name}_test_scores"] = scores
    for split, (_, labels, concurrent) in arrays.items():
        predictions[f"{split}_ids"] = np.array([row["id"] for row in rows[split]])
        predictions[f"{split}_labels"] = labels
        predictions[f"{split}_concurrent_labels"] = concurrent
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    return report, hist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("audit", "pilot", "full"), default="pilot")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=6116)
    parser.add_argument("--weights", type=Path, help="Optional local official ResNet-18 state dictionary")
    args = parser.parse_args()
    if not 1 <= args.frames <= 32 or (args.epochs is not None and args.epochs < 1):
        parser.error("frames must be between 1 and 32 and epochs must be positive")
    args.epochs = args.epochs or (40 if args.mode == "full" else 5)
    if (args.output / "results.json").exists():
        parser.error("Completed experiment exists. Choose a new output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ZipFile(args.archive) as archive:
        data, metadata = load_dataset(archive)
        save_json(args.output / "dataset_audit.json", metadata)
        print("Audit passed:", {split: len(items) for split, items in data.items()})
        if args.mode == "audit":
            return

        import torch
        import torchvision
        import cv2

        torch.set_num_threads(4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = data
        if args.mode == "pilot":
            sampled = random.Random(args.seed).sample(data["train"], 256)
            rows = {"train": sorted(sampled, key=lambda item: item["id"]), "valid": data["valid"]}
        print(f"Running {args.mode} on {device}: {[(s, len(r)) for s, r in rows.items()]}")
        features = extract_features(archive, [r for items in rows.values() for r in items], args, metadata, device)
    models, history = train_models(features, rows, metadata, args, device)
    save_json(args.output / "results.json", {
        "mode": args.mode,
        "device": str(device),
        "software": {"torch": torch.__version__, "torchvision": torchvision.__version__, "opencv": cv2.__version__},
        "manifest_sha256": metadata["manifest_sha256"],
        "train_videos": len(rows["train"]),
        "validation_videos": len(rows["valid"]),
        "test_videos_evaluated": len(rows.get("test", [])),
        "primary_classes": len(metadata["main_class_ids"]),
        "all_classes": len(metadata["classes"]),
        "frozen_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "frames": args.frames,
        "epochs": args.epochs,
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "models": models,
        "history": history,
    })
    print(f"Results: {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
