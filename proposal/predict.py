"""Local inference for the whole-video ResNet classifiers and template grounding."""

import hashlib
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from .experiment import sample_frames
from .models import build_model, classify
from .grounding import WindowScorer, candidate_windows, select_intervals

ROOT = Path(__file__).resolve().parent
OFFICIAL_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def parse_query(query, animals, classes):
    match = re.fullmatch(r"\s*(?:a|an|the)\s+(.+?)\s+is\s+(.+?)\s*[.]?\s*", query, flags=re.I)
    if not match:
        raise ValueError("Use the template 'A {animal} is {action}.'")
    animal, action = (part.strip().rstrip(".").casefold() for part in match.groups())
    names = {name.casefold(): i for i, name in enumerate(classes)}
    if action not in names:
        raise ValueError("Use an action name from the dataset vocabulary.")
    animal_ids = {name.casefold(): i + 1 for i, name in enumerate(animals)}
    return animal_ids.get(animal, 0), names[action]


class ProposalPredictor:
    def __init__(self, device="cpu", weights=None):
        self.device = device
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        path = Path(weights) if weights else ROOT / "resnet18-f37072fd.pth"
        if weights is None:
            from deployment.final_artifacts import ensure_final_runtime
            ensure_final_runtime()
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != OFFICIAL_SHA:
                raise ValueError("ResNet-18 weights do not match the official checkpoint.")
        self.encoder = resnet18()
        self.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.encoder.fc = torch.nn.Identity()
        self.encoder.requires_grad_(False).eval().to(device)

    @torch.inference_mode()
    def features(self, video, count):
        frames = sample_frames(video, count)
        x = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
        return self.encoder(x)[None]

    @torch.inference_mode()
    def predict_boundaries(self, video, query, checkpoint_path):
        """Explicit experimental boundary inference; never promotes smoke weights."""
        from .boundary_grounding import BoundaryModel, dense_features, decode
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != "boundary-v1" or checkpoint.get("smoke", True):
            raise ValueError("A full boundary-v1 checkpoint is required; smoke weights cannot be deployed.")
        animal, action = parse_query(query, checkpoint["animals"], checkpoint["classes"])
        if animal == 0:
            raise ValueError("Boundary localization requires an animal from the training vocabulary.")
        model = BoundaryModel(len(checkpoint["animals"])).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        value = dense_features(self, video, checkpoint["rate"])
        logits, presence = model(value["features"][None].to(self.device),
                                 torch.tensor([animal], device=self.device),
                                 torch.tensor([action], device=self.device))
        return dict(query=query, intervals=decode(logits[0], presence[0], value["edges"], checkpoint["threshold"]),
                    model="boundary-v1", experimental=True, calibrated=False,
                    sampled_timestamps=value["timestamps"], time_edges=value["edges"],
                    max_time_bin_seconds=float(np.diff(value["edges"]).max()),
                    score_meaning="uncalibrated activity score; precision requires independent validation")

    @torch.inference_mode()
    def predict(self, video, kind="bigru_attention", query=None):
        started = time.perf_counter()
        path = ROOT / "results" / f"{kind}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        head = build_model(kind).to(self.device)
        head.load_state_dict(checkpoint["state_dict"])
        head.eval()
        features = self.features(video, checkpoint["frames"])
        logits = classify(head, features)[0].cpu()
        scores = logits.sigmoid().tolist()
        from animalact2.dashboard.registry import classes
        vocabulary = classes()
        if checkpoint["classes"] != [row["name"] for row in vocabulary]:
            raise ValueError("Checkpoint vocabulary does not match dashboard classes.")
        threshold = torch.tensor(checkpoint["threshold"], dtype=torch.float32).item()
        rows = [dict(row, score=p, above_threshold=p >= threshold,
                     eligible_for_prediction=True, selected=p >= threshold,
                     learning_status="Train_supported") for row, p in zip(vocabulary, scores)]
        ranked = sorted(rows, key=lambda row: (-row["score"], row["class_id"]))
        result = {
            "schema_version": 2, "video_id": Path(video).stem,
            "model_id": "resnet_" + kind, "model_name": "ResNet-18 + " + kind,
            "device": self.device, "logits": logits.tolist(), "class_scores": rows,
            "predicted_labels": [row["name"] for row in ranked if row["selected"]],
            "unsupported_class_ids": [], "feature_shape": list(features.shape),
            "calibrated": False,
            "evaluation_unit": "whole_video",
            "model": kind, "threshold": checkpoint["threshold"],
            "scores": [{"action": name, "score": p, "predicted": p >= checkpoint["threshold"]}
                       for name, p in zip(checkpoint["classes"], scores)],
        }
        if query is not None:
            locator = torch.load(ROOT / "grounding_rank_results" / "grounding.pt", map_location="cpu", weights_only=True)
            source = ROOT / "results" / f"{locator['classification_kind']}.pt"
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != locator["classification_sha256"]:
                    raise ValueError("Grounding feature checkpoint changed.")
            temporal_head = build_model(locator["classification_kind"]).to(self.device)
            temporal_head.load_state_dict(torch.load(source, map_location="cpu", weights_only=True)["state_dict"])
            temporal_head.eval()
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=self.device)
            temporal = temporal_head.temporal(features, torch.tensor([features.shape[1]]), mask)
            pooled, windows = candidate_windows(temporal)
            animal, action = parse_query(query, locator["animals"], locator["classes"])
            scorer = WindowScorer(len(locator["animals"])).to(self.device)
            scorer.load_state_dict(locator["state_dict"])
            scorer.eval()
            logits = scorer(pooled, torch.tensor([animal], device=self.device),
                            torch.tensor([action], device=self.device))[0]
            capture = cv2.VideoCapture(str(video))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
            finally:
                capture.release()
            if duration <= 0:
                raise ValueError("Could not determine video duration.")
            result["grounding"] = {
                "query": query, "unknown_animal": animal == 0,
                "intervals": select_intervals(logits.sigmoid().cpu().numpy(), windows.cpu().numpy() * duration),
                "score_meaning": "query-window match score",
                "time_resolution_seconds": duration / checkpoint["frames"],
            }
        result["inference_time"] = time.perf_counter() - started
        return result
