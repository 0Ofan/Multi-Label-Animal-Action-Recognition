"""Head-only checkpoints with exact protocol/cache/model/RNG provenance."""
import torch
from animalact2.phase1_5.checkpoint import make_checkpoint,head_state,restore_rng,implementation_fingerprints as base_fingerprints
from animalact2.phase1_5.config import config_hash,development_fingerprints
from .support import ROOT,OUT,FULL,PILOT,sha,owned


def fingerprints():
    result=base_fingerprints()
    paths=[OUT/(n+".py") for n in ("support","data","models","checkpoint","trainer")]
    paths+=[PILOT/"heads.py"]
    return dict(result,**{p.relative_to(ROOT).as_posix():sha(p) for p in paths})


def cache_provenance():
    return {n:sha(FULL/n) for n in ("artifact_manifest.json","FINAL_VERIFICATION.json","cache/context.json","full_plan.json","validation_summary.json")}


def create(model,optimizer,schedule,early,policy,config,*,epoch,history,diagnostic,resource_state):
    state=make_checkpoint(model,optimizer,schedule,early,policy,config,epoch=epoch,history=history,diagnostic=diagnostic)
    state.update(input_mode="videomae_full_cache_fp32",cache_provenance=cache_provenance(),first_round_implementation=fingerprints(),resources=resource_state,
                 format_note="All learned head/optimizer/RNG state included; frozen VideoMAE reference recorded, no encoder execution or tensors in training")
    return state


def save(path,state):
    path=owned(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp");torch.save(state,temporary);temporary.replace(path)


def load(path,model,config,*,optimizer=None,schedule=None,early=None,policy=None,restore_random=False,diagnostic=False):
    state=torch.load(owned(path),map_location="cpu",weights_only=True)
    if state["config_SHA256"]!=config_hash(config) or state["diagnostic"]!=diagnostic:raise ValueError("Checkpoint config/diagnostic mismatch")
    assert state["input_mode"]=="videomae_full_cache_fp32" and state["cache_provenance"]==cache_provenance()
    assert state["data_fingerprints"]==development_fingerprints() and state["first_round_implementation"]==fingerprints()
    assert state["backbone"]["sha256"]==model.backbone.weight_info["sha256"]
    assert set(state["head_state"])==set(head_state(model))
    model.load_state_dict(state["head_state"],strict=True)
    for obj,key in ((optimizer,"optimizer"),(schedule,"schedule"),(early,"early_stopping"),(policy,"class_policy")):
        if obj is not None:obj.load_state_dict(state[key])
    if policy is not None:policy.restore_rows(model)
    if restore_random:restore_rng(state["RNG"])
    return state
