"""Train and evaluate a query-conditioned temporal-window scorer on AnimalAct."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn

from .models import build_model
from .experiment import save_json


def temporal_iou(windows, references):
    windows, references = np.asarray(windows), np.asarray(references)
    intersection = np.maximum(
        0, np.minimum(windows[:, None, 1], references[None, :, 1])
        - np.maximum(windows[:, None, 0], references[None, :, 0]),
    )
    union = (windows[:, 1] - windows[:, 0])[:, None] + (
        references[:, 1] - references[:, 0]
    )[None] - intersection
    return (intersection / np.maximum(union, 1e-8)).max(axis=1)


def candidate_windows(features):
    """Mean pool every contiguous time-bin window on a uniform proposal grid."""
    steps = features.shape[1]
    pairs = torch.tensor([(a, b) for a in range(steps) for b in range(a + 1, steps + 1)],
                         device=features.device)
    cumulative = nn.functional.pad(features.cumsum(dim=1), (0, 0, 1, 0))
    pooled = (cumulative[:, pairs[:, 1]] - cumulative[:, pairs[:, 0]]) / (
        pairs[:, 1] - pairs[:, 0]
    )[None, :, None]
    windows = pairs.float() / steps
    geometry = torch.cat((windows, windows[:, 1:] - windows[:, :1]), dim=1)
    return torch.cat((pooled, geometry[None].expand(len(features), -1, -1)), dim=2), windows


class WindowScorer(nn.Module):
    def __init__(self, animals):
        super().__init__()
        self.animal = nn.Embedding(animals + 1, 16, padding_idx=0)
        self.action = nn.Embedding(119, 16)
        self.score = nn.Sequential(nn.Linear(256 + 3 + 32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, windows, animals, actions):
        query = torch.cat((self.animal(animals), self.action(actions)), dim=-1)
        return self.score(torch.cat((windows, query[:, None].expand(-1, windows.shape[1], -1)), dim=-1)).squeeze(-1)


def read_queries(archive, ids, classes, animals=None):
    samples = json.loads(archive.read("AnimalAct/manifest.json"))["samples"]
    sources = {row["video_id"]: row for row in samples}
    class_ids = {name: index for index, name in enumerate(classes)}
    collected = {}
    for split, video_ids in ids.items():
        queries = []
        for index, video_id in enumerate(video_ids):
            source = sources[video_id]
            groups = defaultdict(list)
            record = json.loads(archive.read("AnimalAct/" + source["annotation_path"]))
            for animal in record["annotations"]:
                for item in animal["actions"]:
                    start = max(0, float(item["start"]))
                    end = min(source["duration"], float(item["end"]))
                    if end > start:
                        groups[(animal["animal"], class_ids[item["action"]])].append([start, end])
            for (animal, action), spans in sorted(groups.items()):
                queries.append({
                    "index": index, "video_id": video_id, "animal": animal, "action": action,
                    "query": f"A {animal.lower()} is {classes[action].lower()}.",
                    "duration": source["duration"], "references": spans,
                })
        collected[split] = queries
    if animals is None:
        animals = sorted({q["animal"] for q in collected["train"]})
    animal_ids = {name: index + 1 for index, name in enumerate(animals)}
    for queries in collected.values():
        for query in queries:
            query["animal_id"] = animal_ids.get(query["animal"], 0)
    return collected, animals


@torch.inference_mode()
def encode_temporal(head, features, device):
    output = []
    head.eval().to(device)
    for start in range(0, len(features), 32):
        x = torch.from_numpy(features[start:start + 32]).to(device)
        mask = torch.ones(x.shape[:2], device=device, dtype=torch.bool)
        lengths = torch.full((len(x),), x.shape[1], dtype=torch.int64)
        output.append(head.temporal(x, lengths, mask).cpu())
    return torch.cat(output)


def query_tensors(queries, windows, device):
    video_indices = torch.tensor([q["index"] for q in queries], device=device)
    animal_ids = torch.tensor([q["animal_id"] for q in queries], device=device)
    action_ids = torch.tensor([q["action"] for q in queries], device=device)
    overlaps = np.stack([
        temporal_iou(windows * q["duration"], q["references"]) for q in queries
    ]).astype("float32")
    return video_indices, animal_ids, action_ids, torch.from_numpy(overlaps).to(device)


@torch.inference_mode()
def rank_queries(model, candidates, tensors):
    model.eval()
    indices, animals, actions, _ = tensors
    output = []
    for part in torch.arange(len(indices), device=indices.device).split(128):
        output.append(model(candidates[indices[part]], animals[part], actions[part]).cpu())
    return torch.cat(output).numpy()


def evaluate_queries(queries, scores, windows):
    records = []
    for query, score in zip(queries, scores):
        seconds = windows * query["duration"]
        overlaps = temporal_iou(seconds, query["references"])
        top = int(np.argmax(score))
        full = float(temporal_iou(np.array([[0., query["duration"]]]), query["references"])[0])
        records.append({
            **query, "prediction": seconds[top].tolist(), "iou": float(overlaps[top]),
            "whole_video_iou": full, "oracle_candidate_iou": float(overlaps.max()),
            "unknown_animal": query["animal_id"] == 0,
        })
    def metrics(items):
        if not items:
            return {"queries": 0, "miou": None, "r1_iou_05": None, "r1_iou_07": None}
        return {
            "queries": len(items), "miou": float(np.mean([q["iou"] for q in items])),
            "r1_iou_05": float(np.mean([q["iou"] >= .5 for q in items])),
            "r1_iou_07": float(np.mean([q["iou"] >= .7 for q in items])),
            "whole_video_baseline_miou": float(np.mean([q["whole_video_iou"] for q in items])),
            "oracle_candidate_miou": float(np.mean([q["oracle_candidate_iou"] for q in items])),
        }
    return {
        "all": metrics(records),
        "temporally_selective": metrics([q for q in records if q["whole_video_iou"] < .5]),
        "unknown_animal_queries": sum(q["unknown_animal"] for q in records),
        "records": records,
    }


def select_intervals(scores, windows, max_intervals=3):
    chosen = []
    for index in np.argsort(-scores):
        if chosen and temporal_iou(windows[index:index + 1], windows[chosen])[0] > .5:
            continue
        chosen.append(int(index))
        if len(chosen) >= max_intervals:
            break
    return [{"interval": windows[i].tolist(), "score": float(scores[i])} for i in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--experiment", type=Path, default=Path("proposal/results"))
    parser.add_argument("--output", type=Path, default=Path("proposal/grounding_results"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--objective", choices=("iou_bce", "query_ranking"), default="query_ranking")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(6116)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = json.loads((args.experiment / "results.json").read_text())
    # Classification selection is based exclusively on validation mAP.
    kind = max(report["models"], key=lambda name: report["models"][name]["validation"]["map"])
    path = args.experiment / f"{kind}.pt"
    head_state = torch.load(path, map_location="cpu", weights_only=True)
    if kind == "mean":
        kind = max(("bigru_mean", "bigru_attention"),
                   key=lambda name: report["models"][name]["validation"]["map"])
        path = args.experiment / f"{kind}.pt"
        head_state = torch.load(path, map_location="cpu", weights_only=True)
    head = build_model(kind)
    head.load_state_dict(head_state["state_dict"])
    with np.load(args.experiment / "predictions.npz", allow_pickle=False) as data:
        ids = {split: data[f"{split}_ids"].tolist() for split in ("train", "valid", "test")}
    with np.load(args.experiment / "features.npz", allow_pickle=False) as cache:
        config = json.loads(cache["key"].item())
        assert config["video_ids"] == sum(ids.values(), [])
        features = cache["features"]
    with ZipFile(args.archive) as archive:
        queries, animals = read_queries(archive, ids, head_state["classes"])
    temporal = encode_temporal(head, features, device).to(device)
    candidates, windows = candidate_windows(temporal)
    windows_np = windows.cpu().numpy()
    arrays = {}
    offset = 0
    for split in ids:
        end = offset + len(ids[split])
        tensors = query_tensors(queries[split], windows_np, device)
        arrays[split] = (candidates[offset:end], tensors)
        offset = end
    model = WindowScorer(len(animals)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_c, train_t = arrays["train"]
    history = []
    best = -1
    args.output.mkdir(parents=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in torch.randperm(len(queries["train"]), device=device).split(64):
            index, animal, action, overlaps = (v[batch] for v in train_t)
            logits = model(train_c[index], animal, action)
            if args.objective == "iou_bce":
                loss = nn.functional.binary_cross_entropy_with_logits(logits, overlaps)
            else:
                # Reward high-IoU candidates within each query, including repeated occurrences.
                target = overlaps.pow(4)
                target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
                loss = -(target * logits.log_softmax(dim=1)).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scores = rank_queries(model, *arrays["valid"])
        valid = evaluate_queries(queries["valid"], scores, windows_np)
        miou = valid["all"]["miou"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "valid_miou": miou})
        print(f"Grounding epoch {epoch}: loss={np.mean(losses):.4f}, valid mIoU={miou:.4f}", flush=True)
        if miou > best:
            best = miou
            best_epoch = epoch
            state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    with path.open("rb") as file:
        sha = hashlib.file_digest(file, "sha256").hexdigest()
    torch.save({
        "state_dict": state, "animals": animals, "classes": head_state["classes"],
        "classification_kind": kind, "classification_sha256": sha, "frames": head_state["frames"],
        "epoch": best_epoch, "query_format": "A {animal} is {action}.",
        "objective": args.objective,
    }, args.output / "grounding.pt")
    model.load_state_dict(state)
    result = {
        "selection_split": "valid", "selected_epoch": best_epoch, "history": history,
        "classification_kind": kind, "classification_sha256": sha,
        "candidate_count": len(windows), "query_format": "A {animal} is {action}.",
        "objective": args.objective,
        "multiple_reference_rule": "maximum temporal IoU over all matching intervals",
        "validation": evaluate_queries(queries["valid"], rank_queries(model, *arrays["valid"]), windows_np),
        "test": evaluate_queries(queries["test"], rank_queries(model, *arrays["test"]), windows_np),
    }
    save_json(args.output / "results.json", result)
    print(json.dumps({s: result[s]["all"] for s in ("validation", "test")}), flush=True)


if __name__ == "__main__":
    main()
