"""Numerical equivalence and bounded one-pilot-epoch diagnostics for both inputs."""
from __future__ import annotations
import gc
import io
import json
import time
import unittest
import cv2
import torch
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from animalact2.phase1_5_1.collate import move_batch_to_device
from animalact2.phase1_5.config import load_config
from animalact2.phase1_5.data import training_class_support
from animalact2.phase1_5.models import build_model
from animalact2.phase1_5.engine import make_optimizer,train_one_epoch,evaluate_loader
from animalact2.phase1_5.losses import MultilabelLoss,UnsupportedClassPolicy
from animalact2.phase1_5.schedule import LearningRatePlan
from .cache import CACHE,select_plan,selected_records,read_array,clip_descriptor
from .dataset import CachedFeatureDataset,FeatureForwardView,make_pilot_loader
from .support import ROOT,OUT,enable_guard,write_json,sha,digest,RAMMonitor
from . import tests


def state_sha(module):
    import hashlib
    h=hashlib.sha256()
    for name,tensor in module.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def head_state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def cleanup(*objects):gc.collect();torch.cuda.empty_cache()


def main():
    begin=time.perf_counter();torch.set_num_threads(4);cv2.setNumThreads(1);enable_guard()
    if (OUT/"validation_results.json").exists():raise FileExistsError("Keep completed pilot evidence")
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    (OUT/"test_results.txt").write_text(stream.getvalue(),encoding="utf-8")
    if not result.wasSuccessful():print(stream.getvalue());raise RuntimeError("Cache integrity tests failed")
    plan=select_plan();records=selected_records(plan)
    full_train=ReleaseClipDataset("train");counts=training_class_support(full_train)
    raw={s:ReleaseClipDataset(s,clip_ids=plan["selections"][s]) for s in ("train","valid")}
    cached={s:CachedFeatureDataset(s) for s in ("train","valid")}
    cache_before={p.relative_to(CACHE).as_posix():sha(p) for p in CACHE.rglob("*") if p.is_file()}
    configs={name:load_config(ROOT/f"animalact2/phase1_5/configs/{name}_seed42_bce.json") for name in ("model1","model2","model3")}
    # All 120 clips, same untouched task head per model, before any diagnostic updates.
    comparisons=[];gradient_checks=[]
    with RAMMonitor() as comparison_ram:
        for name,config in configs.items():
            model=build_model(config).to("cuda").eval();maximum_logits=0.;maximum_features=0.;compared=0
            first_pair=None;bb_before=state_sha(model.backbone)
            for split in ("train","valid"):
                live_loader=make_pilot_loader(raw[split],config,training=False if split=="valid" else True,cached=False,workers=0)
                cache_loader=make_pilot_loader(cached[split],config,training=False if split=="valid" else True,cached=True,workers=0)
                for a,b in zip(live_loader,cache_loader):
                    assert a["clip_ids"]==b["clip_ids"] and a["parent_video_ids"]==b["parent_video_ids"] and a["splits"]==b["splits"]
                    for k in ("target","lengths","valid_frame_mask","frame_indices"):
                        assert torch.equal(a[k],b[k]),k
                    assert a["label_sources"]==b["label_sources"] and torch.equal(b["padding_mask"],~b["valid_frame_mask"])
                    if first_pair is None and split=="train":first_pair=(a,b)
                    a,b=move_batch_to_device(a,"cuda"),move_batch_to_device(b,"cuda")
                    with torch.no_grad():
                        actual=model(a["frames"],a["lengths"],a["valid_frame_mask"],return_aux=True)
                        reference=model.forward_features(b["frames"],b["lengths"],b["valid_frame_mask"],return_aux=True)
                    error=float((actual["logits"]-reference["logits"]).abs().max());maximum_logits=max(maximum_logits,error)
                    feature_error=float((actual["features"][a["valid_frame_mask"]]-b["frames"][b["valid_frame_mask"]]).abs().max());maximum_features=max(maximum_features,feature_error)
                    torch.testing.assert_close(actual["logits"],reference["logits"],atol=2e-4,rtol=2e-4)
                    if name=="model3":
                        w=reference["temporal_attention"]
                        assert torch.all(w.masked_select(b["padding_mask"][:,None,:].expand_as(w))==0)
                        torch.testing.assert_close(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-6,rtol=0)
                    compared+=len(a["clip_ids"])
            assert compared==120 and state_sha(model.backbone)==bb_before
            # Paired gradients from one real Train batch; no update here.
            criterion=MultilabelLoss(config["loss"],counts>0).to("cuda")
            a,b=[move_batch_to_device(v,"cuda") for v in first_pair]
            gradients=[];losses=[]
            for mode,batch in (("realtime",a),("cached",b)):
                model.train();model.zero_grad(set_to_none=True)
                z=model(batch["frames"],batch["lengths"],batch["valid_frame_mask"]) if mode=="realtime" else model.forward_features(batch["frames"],batch["lengths"],batch["valid_frame_mask"])
                loss=criterion(z,batch["target"]);loss.backward();losses.append(float(loss.detach()))
                grad={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
                assert all(torch.isfinite(g).all() and float(g.norm())>0 for g in grad.values())
                assert all(p.grad is None for p in model.backbone.parameters())
                gradients.append(grad)
            grad_error=max(float((gradients[0][n]-gradients[1][n]).abs().max()) for n in gradients[0])
            assert grad_error<2e-5 and abs(losses[0]-losses[1])<2e-5
            row=dict(model=name,clips_compared=compared,maximum_absolute_logit_error=maximum_logits,maximum_absolute_frame_feature_error=maximum_features,
                     absolute_tolerance=2e-4,relative_tolerance=2e-4,labels_frames_masks_sources_identical=True,backbone_frozen_eval_unchanged=True)
            comparisons.append(row);gradient_checks.append(dict(model=name,maximum_absolute_gradient_difference=grad_error,loss_absolute_difference=abs(losses[0]-losses[1]),all_task_modules_nonzero_finite_gradients=True,backbone_gradients=False))
            print(json.dumps(row),flush=True)
            del model,criterion,a,b,actual,reference,first_pair,gradients,grad;cleanup()
    # Reading benchmark includes per-entry provenance/SHA/shape checks, with no DataLoader workers.
    start=time.perf_counter();payload=0
    with RAMMonitor() as read_ram:
        for dataset in cached.values():
            for i in range(len(dataset)):payload+=dataset[i]["features"].numel()*4
    reading_seconds=time.perf_counter()-start
    read_benchmark=dict(clips=120,FP32_payload_bytes=payload,seconds=reading_seconds,clips_per_second=120/reading_seconds,
                        payload_MiB_per_second=payload/2**20/reading_seconds,SHA_validation_included=True,OS_cache_state="warm after integrity and equivalence checks; not cold disk I/O",RAM=read_ram.result())
    # Identical formal batch/accumulation/LR/WD; only 100+20 pilot clips, one epoch, no checkpoint/selection.
    timings=[];post_states={}
    for name,config in configs.items():
        post_states[name]={}
        for mode in ("realtime","cached"):
            is_cached=mode=="cached";source=cached if is_cached else raw
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            with RAMMonitor() as ram:
                model=build_model(config).to("cuda");policy=UnsupportedClassPolicy(model,counts)
                criterion=MultilabelLoss(config["loss"],policy.active_mask).to("cuda");optimizer=make_optimizer(model,config)
                schedule=LearningRatePlan(optimizer,config);lr=schedule.apply_epoch(1)
                train_loader=make_pilot_loader(source["train"],config,training=True,cached=is_cached)
                valid_loader=make_pilot_loader(source["valid"],config,training=False,cached=is_cached)
                startup=time.perf_counter();warm_a=next(iter(train_loader));warm_b=next(iter(valid_loader));startup=time.perf_counter()-startup
                del warm_a,warm_b
                view=FeatureForwardView(model) if is_cached else model
                initial=head_state(model);bb_before=state_sha(model.backbone)
                torch.cuda.synchronize();t=time.perf_counter()
                trained=train_one_epoch(view,train_loader,optimizer,criterion,policy,"cuda",accumulation_steps=config["data"]["accumulation_steps"],grad_clip_norm=config["optimizer"]["grad_clip_norm"])
                torch.cuda.synchronize();train_seconds=time.perf_counter()-t;t=time.perf_counter()
                evaluated=evaluate_loader(view,valid_loader,criterion,"cuda")
                torch.cuda.synchronize();valid_seconds=time.perf_counter()-t
                assert trained["examples"]==100 and evaluated["examples"]==20 and trained["optimizer_updates"]==4
                current=head_state(model)
                updates={n:float((current[n]-initial[n]).norm()) for n in current}
                assert all(v>0 for v in updates.values())
                assert all(torch.equal(current[n][23],initial[n][23]) for n in current if current[n].shape[0]==119)
                assert state_sha(model.backbone)==bb_before
                post_states[name][mode]=current
            row=dict(model=name,input_mode=mode,scope="one bounded 100Train+20Valid pilot epoch",Train_clips=100,Valid_clips=20,microbatch=8,accumulation=4,optimizer_updates=4,
                     learning_rate_epoch1=lr,worker_count_per_loader=2,worker_startup_seconds=startup,steady_train_seconds=train_seconds,steady_valid_seconds=valid_seconds,
                     steady_pilot_epoch_seconds=train_seconds+valid_seconds,cold_start_pilot_epoch_seconds=startup+train_seconds+valid_seconds,
                     CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),RAM=ram.result(),
                     task_parameter_update_L2=updates,unsupported_row23_unchanged=True,backbone_unchanged=True,
                     Train_order_SHA256=digest(trained["clip_ids"]),Train_target_SHA256=digest(trained["target"].tolist()),Valid_order_SHA256=digest(evaluated["clip_ids"]),
                     no_early_stopping_or_checkpoint_selection=True,formal_config_SHA256=digest(config))
            timings.append(row);print(json.dumps({k:row[k] for k in ("model","input_mode","steady_pilot_epoch_seconds","worker_startup_seconds","CUDA_peak_allocated_bytes")}),flush=True)
            del model,view,policy,criterion,optimizer,schedule,train_loader,valid_loader,trained,evaluated,initial,current;cleanup()
    pairs=[]
    for name in configs:
        a,b=[next(r for r in timings if r["model"]==name and r["input_mode"]==m) for m in ("realtime","cached")]
        assert all(a[k]==b[k] for k in ("Train_order_SHA256","Train_target_SHA256","Valid_order_SHA256","formal_config_SHA256"))
        max_param_error=max(float((post_states[name]["realtime"][n]-post_states[name]["cached"][n]).abs().max()) for n in post_states[name]["realtime"])
        full_live=a["steady_train_seconds"]/100*4667+a["steady_valid_seconds"]/20*979
        full_cache=b["steady_train_seconds"]/100*4667+b["steady_valid_seconds"]/20*979
        pairs.append(dict(model=name,pilot_steady_epoch_speedup=a["steady_pilot_epoch_seconds"]/b["steady_pilot_epoch_seconds"],
                          full_Train_Valid_epoch_realtime_estimate_seconds=full_live,full_Train_Valid_epoch_cached_estimate_seconds=full_cache,
                          estimated_full_epoch_speedup=full_live/full_cache,post_4step_maximum_parameter_difference=max_param_error))
    assert cache_before=={p.relative_to(CACHE).as_posix():sha(p) for p in CACHE.rglob("*") if p.is_file()},"Diagnostics changed immutable cache"
    extraction=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT/"extraction_runs").glob("run_*.json"))]
    assert sum(r["new_unique_frames_inferred"] for r in extraction)==extraction[-1]["unique_frames_expected"]==1699
    assert extraction[-1]["new_unique_frames_inferred"]==0 and extraction[-1]["decoded_parent_videos"]==0 and extraction[-1]["reused_clips"]==120
    bank_files=[p for p in (CACHE/"frames").rglob("*") if p.is_file()];clip_files=[p for p in (CACHE/"clips").rglob("*") if p.is_file()]
    full=plan["full_Train_Valid_metadata_only"];pilot_references=sum(r["effective_length"] for r in records)
    creation_seconds=sum(r["elapsed_seconds"] for r in extraction if r["new_unique_frames_inferred"])
    bank_bytes=sum(p.stat().st_size for p in bank_files);clip_bytes=sum(p.stat().st_size for p in clip_files)
    estimate=dict(scope="full Train4667+Valid979 only; no Test",unique_frames=full["unique_frames"],clip_frame_references=full["clip_frame_references"],
                  unique_frame_FP32_payload_bytes=full["unique_frames"]*512*4,clip_FP32_payload_bytes=full["clip_frame_references"]*512*4,
                  estimated_bank_plus_clip_files_bytes=bank_bytes/1699*full["unique_frames"]+clip_bytes/120*full["clips"],
                  estimated_generation_seconds_by_unique_frames=creation_seconds/1699*full["unique_frames"],
                  generation_uncertainty_multiplier=[.7,2.],capacity_note="Features plus receipts/NPY headers extrapolated; directory/cluster allocation may be larger",
                  estimated_three_models_50epoch_realtime_hours=sum(r["full_Train_Valid_epoch_realtime_estimate_seconds"] for r in pairs)*50/3600,
                  estimated_three_models_50epoch_cached_hours=sum(r["full_Train_Valid_epoch_cached_estimate_seconds"] for r in pairs)*50/3600,
                  future_generation_not_executed=True)
    result=dict(status="passed",clips=dict(train=100,valid=20),unique_parent_videos=len(plan["source_stats"]),unique_frames=1699,clip_frame_references=pilot_references,
                repeated_frame_references_saved=pilot_references-1699,automated_tests=result.testsRun,extraction_runs=extraction,
                logits_comparisons=comparisons,gradient_comparisons=gradient_checks,comparison_RAM=comparison_ram.result(),cache_read_benchmark=read_benchmark,
                pilot_epoch_timings=timings,speedup_pairs=pairs,full_cache_estimate=estimate,cache_bank_file_bytes=bank_bytes,cache_clip_file_bytes=clip_bytes,
                cache_file_count=len(cache_before),cache_total_file_bytes=sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file()),
                raw_or_temporal_features_saved=False,only_ResNet18_FP32_outputs=True,formal_config_or_model_modified=False,formal_training=False,Test_read=False,
                total_validation_seconds=time.perf_counter()-begin,stopped_for_review=True)
    write_json(OUT/"validation_results.json",result)
    print(json.dumps(dict(status="passed",all120_logits_compared_per_model=True,only_pilot_epochs=True,speedups=pairs,total_seconds=result["total_validation_seconds"])),flush=True)


if __name__=="__main__":main()
