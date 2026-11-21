"""Mandatory bounded real-batch validation before any formal epoch."""
from __future__ import annotations
import copy
import gc
import json
import math
import random
import time
import numpy as np
import torch
from animalact2.phase1_5.config import load_config,config_hash
from animalact2.phase1_5.data import training_class_support
from animalact2.phase1_5.engine import train_one_epoch,evaluate_loader,summarize
from animalact2.phase1_5.evaluation import evaluate_outputs,select_validation_threshold
from animalact2.phase1_5.checkpoint import head_state
from animalact2.phase1_5_cache_full.dataset import CachedFeatureDataset
from .trainer import FullCacheTrainer
from . import checkpoint
from .support import ROOT,OUT,enable_guard,write_json


def subset(dataset,lengths):
    ids=[]
    for n in lengths:
        r=next(r for r in dataset.records if r["effective_length"]==n and r["clip_id"] not in ids);ids.append(r["clip_id"])
    return CachedFeatureDataset(dataset.split,clip_ids=ids)


def main():
    enable_guard();torch.set_num_threads(4);started=time.perf_counter()
    if (OUT/"integration_validation.json").exists():raise FileExistsError("Keep passed integration evidence")
    cache=json.loads((OUT/"preflight/cache_and_protocol.json").read_text(encoding="utf-8"));assert cache["status"]=="passed"
    full_train=CachedFeatureDataset("train");full_valid=CachedFeatureDataset("valid");counts=training_class_support(full_train)
    training=subset(full_train,[1,2,3,4,5,10,15,16,16,16,16]);valid=subset(full_valid,[1,3,4,10,15,16])
    checks=[]
    for name in ("model1","model2","model3"):
        config=load_config(ROOT/f"animalact2/phase1_5/configs/{name}_seed42_bce.json")
        trainer=FullCacheTrainer(config,OUT/f"diagnostics/{name}",train_dataset=training,valid_dataset=valid,diagnostic=True,counts=counts,workers_override=0)
        trainer.schedule.apply_epoch(1);trainer.train_loader.sampler.set_epoch(1)
        initial=head_state(trainer.model)
        out=train_one_epoch(trainer.view,trainer.train_loader,trainer.optimizer,trainer.criterion,trainer.policy,"cuda",accumulation_steps=4,grad_clip_norm=5.)
        evaluated=evaluate_loader(trainer.view,trainer.valid_loader,trainer.criterion,"cuda")
        assert out["examples"]==11 and out["optimizer_updates"]==1 and evaluated["examples"]==6
        metrics=summarize(evaluated,trainer.policy,full=False)
        assert metrics["main_mAP"] is None and math.isfinite(metrics["BCE119"])
        updated=head_state(trainer.model)
        assert all(float((updated[n]-initial[n]).norm())>0 for n in updated)
        assert all(torch.equal(updated[n][23],initial[n][23]) for n in updated if updated[n].shape[0]==119)
        state=trainer.make_state(0,[],dict(validation_only=True))
        path=OUT/f"diagnostics/{name}/roundtrip.pt";checkpoint.save(path,state)
        expected_rng=(random.random(),float(np.random.rand()),torch.rand(4))
        # A fresh independent model/optimizer, matching the actual formal resume path.
        restored=FullCacheTrainer(config,OUT/f"diagnostics/{name}_reload",train_dataset=training,valid_dataset=valid,diagnostic=True,counts=counts,workers_override=0)
        checkpoint.load(path,restored.model,config,optimizer=restored.optimizer,schedule=restored.schedule,early=restored.early,policy=restored.policy,restore_random=True,diagnostic=True)
        got_rng=(random.random(),float(np.random.rand()),torch.rand(4))
        assert expected_rng[:2]==got_rng[:2] and torch.equal(expected_rng[2],got_rng[2])
        after=evaluate_loader(restored.view,restored.valid_loader,restored.criterion,"cuda")
        torch.testing.assert_close(evaluated["logits"],after["logits"],atol=0,rtol=0)
        assert restored.schedule.last_epoch==1 and restored.optimizer.state_dict()["state"]
        rejected=False
        try:checkpoint.load(path,restored.model,config,diagnostic=False)
        except ValueError:rejected=True
        assert rejected and trainer.CNN_calls==0 and restored.CNN_calls==0
        # Valid-only threshold/31-class AP machinery on an explicit artificial fixture.
        z=torch.full((2,119),-10.);y=torch.zeros(2,119);z[0]=10.;y[0]=1.
        fixture=evaluate_outputs(z,y,["fixture0","fixture1"],trainer.policy)
        assert fixture["main_mAP"]==fixture["main_macro_f1"]==1.
        assert select_validation_threshold(z,y,split="valid")["threshold"]==.05
        assert fixture["per_class"][23]["eligible_for_prediction"] is False
        checks.append(dict(model=name,status="passed",Train_clips=11,Valid_clips=6,partial_real_Valid_mAP=None,real_loss=out["objective"],real_Valid_BCE119=metrics["BCE119"],
                           backward_and_head_update=True,row23_fixed=True,CNN_calls=0,checkpoint_logits_exact=True,optimizer_and_rng_restored=True,diagnostic_resume_into_formal_rejected=True,
                           fixed31_metric_fixture_passed=True))
        trainer.close();restored.close();del trainer,restored,state,initial,updated,out,evaluated,after;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(checks[-1],ensure_ascii=False),flush=True)
    result=dict(status="passed",models=checks,cache_provenance=checkpoint.cache_provenance(),integration_fingerprints=checkpoint.fingerprints(),
                config_SHA256=cache["config_SHA256"],shared_cohort_SHA256=cache["shared_cohort_SHA256"],seconds=time.perf_counter()-started,
                real_diagnostic_updates=3,formal_training_started=False,Test_read=False,ASL_used=False)
    write_json(OUT/"integration_validation.json",result);print(json.dumps(dict(integration="passed",seconds=result["seconds"])),flush=True)


if __name__=="__main__":main()
