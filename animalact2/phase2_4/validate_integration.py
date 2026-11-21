"""Bounded Train updates, full real Valid metrics, masking and exact checkpoint restore."""
import gc
import json
import math
import random
import time
import numpy as np
import torch
from animalact2.phase1_5.config import config_hash
from animalact2.phase1_5.data import training_class_support
from animalact2.phase1_5.engine import train_one_epoch,evaluate_loader,summarize
from animalact2.phase1_5.checkpoint import head_state
from animalact2.phase1_5.evaluation import select_validation_threshold
from .models import load_config
from .data import CachedFeatureDataset,collate_features
from .trainer import FullCacheTrainer
from . import checkpoint
from .support import OUT,enable_guard,write_json


def main():
    enable_guard();torch.set_num_threads(4);started=time.perf_counter()
    if (OUT/"integration_validation.json").exists():raise FileExistsError("Keep passed diagnostic evidence")
    gate=json.loads((OUT/"preflight/cache_and_protocol.json").read_text(encoding="utf-8"));assert gate["status"]=="passed"
    full_train=CachedFeatureDataset("train");full_valid=CachedFeatureDataset("valid");counts=training_class_support(full_train)
    ids=[]
    for length in (1,2,3,4,5,10,15,16,16,16,16):
        ids.append(next(r["clip_id"] for r in full_train.records if r["effective_length"]==length and r["clip_id"] not in ids))
    training=CachedFeatureDataset("train",clip_ids=ids)
    mixed=collate_features([training[i] for i in (0,2,4,7)])
    assert mixed["frames"].shape==(4,8,768) and mixed["lengths"].tolist()==[1,2,3,8]
    checks=[]
    for name in ("model1","model2","model3"):
        config=load_config(OUT/f"configs/{name}_seed42_bce.json")
        trainer=FullCacheTrainer(config,OUT/f"diagnostics/{name}",train_dataset=training,valid_dataset=full_valid,diagnostic=True,counts=counts,workers_override=0)
        trainer.schedule.apply_epoch(1);trainer.train_loader.sampler.set_epoch(1)
        initial=head_state(trainer.model)
        training_output=train_one_epoch(trainer.view,trainer.train_loader,trainer.optimizer,trainer.criterion,trainer.policy,"cuda",accumulation_steps=4,grad_clip_norm=5.)
        evaluated=evaluate_loader(trainer.view,trainer.valid_loader,trainer.criterion,"cuda")
        assert training_output["examples"]==11 and training_output["optimizer_updates"]==1 and evaluated["examples"]==979
        metrics=summarize(evaluated,trainer.policy)
        assert metrics["main_mAP"] is not None and math.isfinite(metrics["main_mAP"]) and math.isfinite(metrics["BCE119"])
        threshold=select_validation_threshold(evaluated["logits"],evaluated["target"],split="valid")
        assert threshold["split"]=="valid" and .05<=threshold["threshold"]<=.95
        updated=head_state(trainer.model)
        assert all(float((updated[k]-initial[k]).norm())>0 for k in updated)
        for k in trainer.policy.anchors:assert torch.equal(updated[k][23],initial[k][23])
        trainer.model.eval()
        with torch.no_grad():
            features=mixed["frames"].cuda();mask=mixed["valid_frame_mask"].cuda()
            a=trainer.model(features,mixed["lengths"],mask,return_aux=True)
            poisoned=features.clone();poisoned[~mask]=12345.
            b=trainer.model(poisoned,mixed["lengths"],mask)
            assert torch.equal(a["logits"],b)
            if name=="model3":
                assert torch.all(a["temporal_attention"].masked_select(~mask[:,None,:])==0)
                torch.testing.assert_close(a["temporal_attention"].sum(-1),torch.ones(4,119,device="cuda"))
        state=trainer.make_state(0,[],dict(validation_only=True));path=OUT/f"diagnostics/{name}/roundtrip.pt";checkpoint.save(path,state)
        expected=(random.random(),float(np.random.rand()),torch.rand(4),torch.rand(4,device="cuda").cpu())
        restored=FullCacheTrainer(config,OUT/f"diagnostics/{name}_reload",train_dataset=training,valid_dataset=full_valid,diagnostic=True,counts=counts,workers_override=0)
        checkpoint.load(path,restored.model,config,optimizer=restored.optimizer,schedule=restored.schedule,early=restored.early,policy=restored.policy,restore_random=True,diagnostic=True)
        got=(random.random(),float(np.random.rand()),torch.rand(4),torch.rand(4,device="cuda").cpu())
        assert expected[:2]==got[:2] and all(torch.equal(x,y) for x,y in zip(expected[2:],got[2:]))
        after=evaluate_loader(restored.view,restored.valid_loader,restored.criterion,"cuda")
        torch.testing.assert_close(evaluated["logits"],after["logits"],atol=0,rtol=0)
        assert restored.schedule.last_epoch==1 and restored.optimizer.state_dict()["state"]
        rejected=False
        try:checkpoint.load(path,restored.model,config,diagnostic=False)
        except ValueError:rejected=True
        assert rejected and trainer.encoder_calls==restored.encoder_calls==0
        checks.append(dict(model=name,status="passed",Train_diagnostic_clips=11,Valid_clips=979,diagnostic_optimizer_updates=1,
                           output_shape=[979,119],diagnostic_Valid_mAP=metrics["main_mAP"],diagnostic_Valid_BCE119=metrics["BCE119"],
                           finite_real_metrics=True,Valid_threshold_selection_passed=True,all_head_parameters_updated=True,row23_unchanged=True,
                           checkpoint_full_Valid_logits_exact=True,optimizer_schedule_and_all_RNG_restored=True,diagnostic_resume_into_formal_rejected=True,
                           mixed_feature_lengths=[1,2,3,8],padding_invariance=True,encoder_calls=0))
        trainer.close();restored.close()
        del trainer,restored,state,initial,updated,training_output,evaluated,after,features,mask,a,b,poisoned;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(checks[-1]),flush=True)
    result=dict(status="passed",models=checks,cache_provenance=checkpoint.cache_provenance(),integration_fingerprints=checkpoint.fingerprints(),
                config_SHA256=gate["config_SHA256"],shared_cohort_SHA256=gate["shared_cohort_SHA256"],seconds=time.perf_counter()-started,
                real_diagnostic_updates=3,formal_training_started=False,Test_read=False,ASL_used=False,
                diagnostic_scores_are_not_formal_results=True)
    write_json(OUT/"integration_validation.json",result);print(json.dumps(dict(integration="passed",seconds=result["seconds"])),flush=True)


if __name__=="__main__":main()
