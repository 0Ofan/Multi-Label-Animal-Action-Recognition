"""Scoped wrapper of the validated checkpoint format, with full-cache provenance."""
import json
import torch
from animalact2.phase1_5.checkpoint import make_checkpoint,head_state,restore_rng,implementation_fingerprints as base_fingerprints
from animalact2.phase1_5.config import config_hash,development_fingerprints
from .support import ROOT,OUT,FULL,owned,sha


def fingerprints():
    result=base_fingerprints()
    for name in ("checkpoint","trainer","support"):
        p=OUT/(name+".py");result[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ("dataset.py","storage.py"):
        p=FULL/name;result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def cache_provenance():
    return {name:sha(FULL/name) for name in ("artifact_manifest.json","cache/context.json","full_plan.json","validation_summary.json")}


def create(model,optimizer,schedule,early,policy,config,*,epoch,history,diagnostic,resource_state):
    result=make_checkpoint(model,optimizer,schedule,early,policy,config,epoch=epoch,history=history,diagnostic=diagnostic)
    result.update(input_mode="shared_full_cache_fp32",cache_provenance=cache_provenance(),first_round_implementation=fingerprints(),resources=resource_state)
    return result


def save(path,state):
    path=owned(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=owned(path.with_suffix(path.suffix+".tmp"));torch.save(state,temporary);temporary.replace(path)


def load(path,model,config,*,optimizer=None,schedule=None,early=None,policy=None,restore_random=False,diagnostic=False):
    state=torch.load(owned(path),map_location="cpu",weights_only=True)
    if state["config_SHA256"]!=config_hash(config) or state["diagnostic"]!=diagnostic:raise ValueError("Checkpoint config/diagnostic mismatch")
    if state["input_mode"]!="shared_full_cache_fp32" or state["cache_provenance"]!=cache_provenance():raise ValueError("Full Cache provenance changed")
    if state["data_fingerprints"]!=development_fingerprints() or state["first_round_implementation"]!=fingerprints():raise ValueError("Data or implementation changed on resume")
    if state["backbone"]["sha256"]!=model.backbone.weight_info["sha256"]:raise ValueError("Backbone initialization changed")
    if set(state["head_state"])!=set(head_state(model)):raise ValueError("Different head structure")
    result=model.load_state_dict(state["head_state"],strict=False)
    assert not result.unexpected_keys and all(k.startswith("backbone.") for k in result.missing_keys)
    for obj,key in ((optimizer,"optimizer"),(schedule,"schedule"),(early,"early_stopping"),(policy,"class_policy")):
        if obj is not None:obj.load_state_dict(state[key])
    if policy is not None:policy.restore_rows(model)
    if restore_random:restore_rng(state["RNG"])
    return state
