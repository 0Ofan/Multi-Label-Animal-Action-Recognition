"""Learn query-conditioned time-bin activity before ranking temporal windows."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn

from .experiment import save_json
from .grounding import read_queries, encode_temporal, candidate_windows, evaluate_queries
from .models import build_model


class DenseScorer(nn.Module):
    def __init__(self, animals):
        super().__init__()
        self.animal = nn.Embedding(animals + 1, 16, padding_idx=0)
        self.action = nn.Embedding(119, 16)
        self.network = nn.Sequential(nn.Linear(289, 128), nn.ReLU(), nn.Dropout(.1),
                                     nn.Linear(128, 1))

    def forward(self, x, animal, action):
        query = torch.cat((self.animal(animal), self.action(action)), -1)
        time = torch.linspace(0, 1, x.shape[1], device=x.device)[None, :, None].expand(len(x), -1, -1)
        return self.network(torch.cat((x, query[:, None].expand(-1, x.shape[1], -1), time), -1)).squeeze(-1)


def activity_target(query, steps):
    """Fraction of each time bin covered by the union of matching intervals."""
    spans = []
    for start, end in sorted(query["references"]):
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(end, spans[-1][1])
        else:
            spans.append([start, end])
    edges = np.linspace(0, query["duration"], steps + 1)
    covered = np.zeros(steps)
    for start, end in spans:
        covered += np.maximum(0, np.minimum(edges[1:], end) - np.maximum(edges[:-1], start))
    return np.clip(covered / np.diff(edges), 0, 1).astype("float32")


def window_scores(logits, windows):
    steps = logits.shape[1]
    inside = torch.stack([
        ((torch.arange(steps, device=logits.device) >= int(round(a * steps)))
         & (torch.arange(steps, device=logits.device) < int(round(b * steps)))).float()
        for a, b in windows.tolist()
    ])
    # Bernoulli log likelihood includes evidence outside the proposed interval.
    return (nn.functional.logsigmoid(logits) @ inside.T
            + nn.functional.logsigmoid(-logits) @ (1 - inside).T)


@torch.inference_mode()
def predict(model, x, queries, windows):
    model.eval()
    outputs = []
    for offset in range(0, len(queries), 128):
        part = queries[offset:offset + 128]
        ids = torch.tensor([q["index"] for q in part], device=x.device)
        animals = torch.tensor([q["animal_id"] for q in part], device=x.device)
        actions = torch.tensor([q["action"] for q in part], device=x.device)
        outputs.append(window_scores(model(x[ids], animals, actions), windows).cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("proposal/dense_grounding_results"))
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(6116)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("proposal/results")
    ck = torch.load(root / "bigru_attention.pt", weights_only=True, map_location="cpu")
    head = build_model("bigru_attention")
    head.load_state_dict(ck["state_dict"])
    with np.load(root / "predictions.npz", allow_pickle=False) as data:
        ids = {s: data[f"{s}_ids"].tolist() for s in ("train", "valid")}
    with ZipFile(args.archive) as archive:
        queries, animals = read_queries(archive, ids, ck["classes"])
    with np.load(root / "features.npz", allow_pickle=False) as cache:
        features = cache["features"]
        temporal = encode_temporal(head, features[:sum(map(len, ids.values()))], device).to(device)
    _, windows = candidate_windows(temporal[:1])
    grid = windows.cpu().numpy()
    x = {"train": temporal[:len(ids["train"])], "valid": temporal[len(ids["train"]):]}
    model = DenseScorer(len(animals)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    train = queries["train"]
    vi = torch.tensor([q["index"] for q in train], device=device)
    ai = torch.tensor([q["animal_id"] for q in train], device=device)
    ci = torch.tensor([q["action"] for q in train], device=device)
    target = torch.from_numpy(np.stack([activity_target(q, ck["frames"]) for q in train])).to(device)
    best, history = -1, []
    args.output.mkdir(parents=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in torch.randperm(len(train), device=device).split(64):
            logits = model(x["train"][vi[batch]], ai[batch], ci[batch])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        result = evaluate_queries(queries["valid"], predict(model, x["valid"], queries["valid"], windows), grid)
        metric = result["all"]["miou"]
        history.append({"epoch": epoch, "validation_miou": metric})
        if metric > best:
            best, selected = metric, epoch
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Dense epoch {epoch}: Valid mIoU {metric:.4f}; best {best:.4f}", flush=True)
    model.load_state_dict(state)
    result = evaluate_queries(queries["valid"], predict(model, x["valid"], queries["valid"], windows), grid)
    baseline = result["all"]["whole_video_baseline_miou"]
    torch.save({"state_dict": state, "animals": animals, "classes": ck["classes"],
                "epoch": selected, "frames": ck["frames"], "classification_kind": "bigru_attention"},
               args.output / "grounding.pt")
    report = {"selection_split": "valid", "selected_epoch": selected, "validation": result,
              "history": history, "test_evaluated": False}
    if best > baseline:
        with np.load(root / "predictions.npz", allow_pickle=False) as data:
            test_ids = {"test": data["test_ids"].tolist()}
        with ZipFile(args.archive) as archive:
            test, _ = read_queries(archive, test_ids, ck["classes"], animals)
        tx = encode_temporal(head, features[sum(map(len, ids.values())):], device).to(device)
        report["test"] = evaluate_queries(test["test"], predict(model, tx, test["test"], windows), grid)
        report["test_evaluated"] = True
    save_json(args.output / "results.json", report)
    print(json.dumps({s: report[s]["all"] for s in ("validation", "test") if s in report}), flush=True)


if __name__ == "__main__":
    main()
