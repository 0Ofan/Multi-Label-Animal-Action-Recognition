"""Small frozen-backbone checkpoints, strict provenance, optimizer/RNG resume."""
from __future__ import annotations
import copy
import random
from pathlib import Path
import numpy as np
import torch
from .config import config_hash, development_fingerprints
from .safety import OUT, ROOT, sha


def owned(path):
    p = Path(path).resolve()
    if not p.is_relative_to(OUT):
        raise ValueError("Checkpoint must belong to phase1_5")
    return p


def implementation_fingerprints():
    previous = ROOT / "animalact2/phase1_5_1"
    old = [previous / (n + ".py") for n in ("dataset", "video_decoder", "transforms", "collate", "common", "backbone", "model1", "model2", "model3", "evaluation")]
    new = [OUT / (n + ".py") for n in ("config", "data", "models", "losses", "schedule", "evaluation", "checkpoint", "engine")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in old+new}


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(name=n[0], keys=n[1].tolist(), position=n[2], has_gauss=n[3], cached_gaussian=n[4]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def head_state(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith("backbone.")}


def make_checkpoint(model, optimizer, schedule, early, policy, config, *, epoch, history, diagnostic):
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("This compact checkpoint format requires an unchanged frozen backbone")
    return dict(schema_version=1, diagnostic=bool(diagnostic), epoch=int(epoch), config=copy.deepcopy(config), config_SHA256=config_hash(config),
                backbone=dict(model.backbone.weight_info), head_state=head_state(model),
                optimizer=optimizer.state_dict(), schedule=schedule.state_dict(), early_stopping=early.state_dict(), class_policy=policy.state_dict(),
                RNG=rng_state(), history=copy.deepcopy(history), data_fingerprints=development_fingerprints(), implementation_fingerprints=implementation_fingerprints(),
                format_note="Frozen backbone reconstructed from verified official local ImageNet weights; all learned head and optimizer state included")


def save_checkpoint(path, checkpoint):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = owned(path.with_suffix(path.suffix + ".tmp"))
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, config, *, optimizer=None, schedule=None, early=None, policy=None, restore_random_state=False, expected_diagnostic=None):
    checkpoint = torch.load(owned(path), map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != 1 or checkpoint["config_SHA256"] != config_hash(config):
        raise ValueError("Checkpoint model/config mismatch")
    if expected_diagnostic is not None and checkpoint["diagnostic"] != expected_diagnostic:
        raise ValueError("Diagnostic checkpoints must not initialize formal experiments")
    if checkpoint["data_fingerprints"] != development_fingerprints() or checkpoint["implementation_fingerprints"] != implementation_fingerprints():
        raise ValueError("Dataset or implementation changed; use a new experiment")
    if checkpoint["backbone"]["sha256"] != model.backbone.weight_info["sha256"]:
        raise ValueError("Different ImageNet backbone initialization")
    if set(checkpoint["head_state"]) != set(head_state(model)):
        raise ValueError("Different model head structure")
    outcome = model.load_state_dict(checkpoint["head_state"], strict=False)
    assert not outcome.unexpected_keys and all(k.startswith("backbone.") for k in outcome.missing_keys)
    for component, key in ((optimizer, "optimizer"), (schedule, "schedule"), (early, "early_stopping"), (policy, "class_policy")):
        if component is not None:
            component.load_state_dict(checkpoint[key])
    if policy is not None:
        policy.restore_rows(model)
    if restore_random_state:
        restore_rng(checkpoint["RNG"])
    return checkpoint


def restore_model(path, *, device="cpu", allow_diagnostic=False):
    """Public model loader; returns unchanged 119-output model and eligibility metadata."""
    from .models import build_model
    from .losses import UnsupportedClassPolicy
    state = torch.load(owned(path), map_location="cpu", weights_only=True)
    model = build_model(state["config"]).to(device)
    policy = UnsupportedClassPolicy(model, state["class_policy"]["train_positive_counts"])
    load_checkpoint(path, model, state["config"], policy=policy,
                    expected_diagnostic=None if allow_diagnostic else False)
    model.eval()
    return model, policy, state
