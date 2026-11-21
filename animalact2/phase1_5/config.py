"""Explicit common experiment protocol, validated before any training."""
from __future__ import annotations
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from animalact2.phase1_5_1.dataset import RELEASE_ID
from .safety import ROOT

BASE = dict(schema_version=1, release_id=RELEASE_ID, model="model1", seed=42, device="cuda",
            backbone=dict(weights="IMAGENET1K_V1", frozen=True, frame_chunk_size=8),
            data=dict(batch_size=8, valid_batch_size=8, accumulation_steps=4, num_workers=2, prefetch_factor=1, pin_memory=True),
            loss=dict(name="bce", gamma_neg=4., gamma_pos=0., clip=.05),
            optimizer=dict(name="AdamW", lr=.001, weight_decay=.0001, grad_clip_norm=5.),
            scheduler=dict(name="warmup_cosine", warmup_epochs=2, min_lr_ratio=.1),
            training=dict(max_epochs=100, patience=15, min_delta=0., precision="fp32", train_metrics="online_preupdate", best_metric="valid_main_mAP"),
            unsupported_class_policy="mask_loss_restore_class_rows_do_not_emit", sampling="each_Train_clip_once_per_epoch_fixed_release_frames")


def default_config(model="model1", seed=42):
    config = copy.deepcopy(BASE)
    config.update(model=model, seed=int(seed))
    return validate_config(config)


def validate_config(c):
    if set(c) != set(BASE):
        raise ValueError("Unknown or missing experiment configuration keys")
    if c["release_id"] != RELEASE_ID or c["model"] not in ("model1", "model2", "model3"):
        raise ValueError("Only release-v1 and the three Proposal models are supported")
    if type(c["seed"]) is not int or c["seed"] < 0 or c["device"] not in ("cpu", "cuda"):
        raise ValueError("Invalid seed/device")
    if c["backbone"]["weights"] != "IMAGENET1K_V1" or c["backbone"]["frozen"] is not True:
        raise ValueError("First-round protocol keeps the same official ImageNet backbone frozen")
    for key in ("batch_size", "valid_batch_size", "accumulation_steps", "prefetch_factor"):
        if type(c["data"][key]) is not int or c["data"][key] < 1:
            raise ValueError("Invalid loader batch/accumulation/prefetch setting")
    if c["data"]["num_workers"] < 0 or c["backbone"]["frame_chunk_size"] < 1:
        raise ValueError("Invalid worker/chunk count")
    if c["loss"]["name"] not in ("bce", "asl") or min(c["loss"]["gamma_neg"], c["loss"]["gamma_pos"]) < 0 or not 0 <= c["loss"]["clip"] < 1:
        raise ValueError("Invalid loss configuration")
    if c["optimizer"]["name"] != "AdamW" or c["optimizer"]["lr"] <= 0 or c["optimizer"]["weight_decay"] < 0 or c["optimizer"]["grad_clip_norm"] <= 0:
        raise ValueError("Invalid optimizer")
    t, schedule = c["training"], c["scheduler"]
    if t["max_epochs"] < 1 or t["patience"] < 1 or t["min_delta"] < 0 or t["precision"] != "fp32" or t["best_metric"] != "valid_main_mAP" or t["train_metrics"] != "online_preupdate":
        raise ValueError("Unsupported training/selection protocol")
    if schedule["name"] not in ("fixed", "warmup_cosine") or not 0 <= schedule["warmup_epochs"] < t["max_epochs"] or not 0 < schedule["min_lr_ratio"] <= 1:
        raise ValueError("Invalid schedule")
    if c["unsupported_class_policy"] != BASE["unsupported_class_policy"] or c["sampling"] != BASE["sampling"]:
        raise ValueError("Do not silently alter eligibility or data sampling protocol")
    return c


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def cohort_hash(config):
    shared = copy.deepcopy(config)
    shared.pop("model")
    return config_hash(shared)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def development_fingerprints():
    from .safety import sha
    base = ROOT / "animalact2/release_v1"
    files = ("version.json", "preprocessing.json", "class_mapping.json", "main_evaluation_classes.json", "train/manifest.jsonl", "valid/manifest.jsonl")
    return {name: sha(base / name) for name in files}
