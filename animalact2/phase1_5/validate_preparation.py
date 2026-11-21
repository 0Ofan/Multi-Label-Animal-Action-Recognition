"""Bounded diagnostics and throughput estimation, never Trainer.run()."""
from __future__ import annotations
import copy
import gc
import hashlib
import io
import json
import math
import time
import unittest
import cv2
import torch
from animalact2.phase1_5_1.tests import test_contracts
from .tests import test_framework
from .data import ReleaseClipDataset, training_class_support, make_loader, collate_clips, move_batch_to_device, EpochShuffleSampler
from .config import default_config, cohort_hash, load_config
from .models import build_model
from .losses import MultilabelLoss, UnsupportedClassPolicy
from .engine import Trainer, train_one_epoch, evaluate_loader, summarize, make_optimizer, write_history
from .checkpoint import save_checkpoint, restore_model, head_state
from .evaluation import save_evaluation
from .safety import OUT, ROOT, enable_guard, write_json


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def module_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def select_lengths(dataset, desired):
    ids, parents = [],set()
    for n in desired:
        candidates=[r for r in dataset.records if r["effective_length"] == n and r["clip_id"] not in ids]
        if not candidates:raise ValueError(f"No diagnostic input of length {n}")
        candidates.sort(key=lambda r:(r["parent_video_id"] in parents,r["clip_id"]))
        chosen=candidates[0];ids.append(chosen["clip_id"]);parents.add(chosen["parent_video_id"])
    return ReleaseClipDataset(dataset.split,clip_ids=ids)


def main():
    started=time.perf_counter();enable_guard();torch.set_num_threads(4);cv2.setNumThreads(1)
    if (OUT/"preparation_validation.json").exists():raise FileExistsError("Preserve completed preparation diagnostics")
    audit=json.loads((OUT/"data_audit/summary.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed" and audit["failures"] == 0
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_contracts,test_framework)])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/"unit_test_results.txt").write_text(stream.getvalue(),encoding="utf-8")
    if not result.wasSuccessful():
        print(stream.getvalue(),flush=True);raise RuntimeError("Unit tests failed")
    print(json.dumps(dict(unit_tests=result.testsRun,status="passed")),flush=True)
    full_train,full_valid=ReleaseClipDataset("train"),ReleaseClipDataset("valid")
    counts=training_class_support(full_train)
    assert (counts == 0).nonzero(as_tuple=False).flatten().tolist() == [23]
    mixed_train=select_lengths(full_train,[1,2,3,5,10,15,16,16])
    mixed_valid=select_lengths(full_valid,[1,3,10,16])
    diagnostic_records=[];all_input_keys=set()
    for model_name in ("model1","model2","model3"):
        for loss_name in ("bce","asl"):
            config=default_config(model_name);config["loss"]["name"]=loss_name
            config["data"].update(num_workers=0,valid_batch_size=4)
            directory=OUT/f"diagnostics/{model_name}_{loss_name}"
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            trainer=Trainer(config,directory,train_dataset=mixed_train,valid_dataset=mixed_valid,diagnostic=True,train_positive_counts=counts)
            trainer.train_loader.sampler.set_epoch(1);trainer.schedule.apply_epoch(1)
            initial=head_state(trainer.model);bb_before=module_sha(trainer.model.backbone)
            training=train_one_epoch(trainer.model,trainer.train_loader,trainer.optimizer,trainer.criterion,trainer.policy,"cuda",accumulation_steps=4,grad_clip_norm=5.,max_batches=1)
            validation=evaluate_loader(trainer.model,trainer.valid_loader,trainer.criterion,"cuda",max_batches=1)
            assert training["examples"] == 8 and validation["examples"] == 4 and training["optimizer_updates"] == 1
            assert len(set(training["clip_ids"])) == 8
            all_input_keys.add((tuple(training["clip_ids"]),tensor_sha(training["target"])))
            changed={k:float((v-initial[k]).norm()) for k,v in head_state(trainer.model).items()}
            assert all(v>0 for v in changed.values())
            assert all(torch.equal(v[23],initial[k][23]) for k,v in head_state(trainer.model).items() if v.ndim and v.shape[0]==119)
            assert module_sha(trainer.model.backbone) == bb_before
            metrics=summarize(validation,trainer.policy,full=False)
            assert metrics["main_mAP"] is None  # no fabricated full31 score from four clips
            save_evaluation(directory/"partial_valid_functional_only",metrics)
            history=[dict(diagnostic=True,optimizer_updates=1,train_objective=training["objective"],valid_objective=validation["objective"],valid_full31_mAP=None)]
            directory.mkdir(parents=True,exist_ok=True);write_history(directory,history)
            save_checkpoint(directory/"diagnostic.pt",trainer.checkpoint(0,history))
            restored,restored_policy,checkpoint=restore_model(directory/"diagnostic.pt",device="cuda",allow_diagnostic=True)
            restored_outputs=evaluate_loader(restored,trainer.valid_loader,trainer.criterion,"cuda",max_batches=1)
            torch.testing.assert_close(restored_outputs["logits"],validation["logits"],atol=0,rtol=0)
            assert restored_policy.metadata() == trainer.policy.metadata() and checkpoint["diagnostic"] is True
            torch.cuda.synchronize()
            record=dict(model=model_name,loss=loss_name,mode="bounded_functional_only",Train_clips=8,Valid_clips=4,optimizer_updates=1,
                        train_lengths=[r["effective_length"] for r in mixed_train.records],valid_lengths=[r["effective_length"] for r in mixed_valid.records],
                        input_clip_ids=training["clip_ids"],target_SHA256=tensor_sha(training["target"]),output_shape=[4,119],
                        training_objective=training["objective"],validation_objective=validation["objective"],all_trainable_tensor_updates_L2=changed,
                        unsupported_row23_unchanged=True,backbone_and_BN_unchanged=True,checkpoint_logits_identical=True,
                        partial_valid_full31_mAP=None,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.perf_counter()-begin)
            diagnostic_records.append(record);print(json.dumps({k:record[k] for k in ("model","loss","optimizer_updates","peak_allocated_bytes","checkpoint_logits_identical")}),flush=True)
            del trainer,restored,restored_policy,training,validation,restored_outputs,initial;gc.collect();torch.cuda.empty_cache()
    assert len(all_input_keys) == 1
    # Representative small sample from the fixed first-epoch order, not performance-selected clips.
    sampler=EpochShuffleSampler(full_train,42);sampler.set_epoch(1)
    bench_train=ReleaseClipDataset("train",clip_ids=[full_train.records[i]["clip_id"] for i in list(sampler)[:32]])
    bench_valid=ReleaseClipDataset("valid",clip_ids=[r["clip_id"] for r in full_valid.records[:16]])
    loader_records=[]
    for workers in (0,2):
        c=default_config();c["data"]["num_workers"]=workers
        loader=make_loader(bench_train,c,training=True);loader.sampler.set_epoch(1)
        start=time.perf_counter();times=[];n=0;ids=[]
        for batch in loader:
            times.append(time.perf_counter()-start);n+=len(batch["clip_ids"]);ids.extend(batch["clip_ids"])
        total=time.perf_counter()-start
        row=dict(workers=workers,clips=n,batches=len(times),first_batch_seconds=times[0],total_seconds=total,
                 steady_seconds_per_clip=(times[-1]-times[0])/(n-8),ordered_clip_ids=ids)
        loader_records.append(row);print(json.dumps({k:v for k,v in row.items() if k!="ordered_clip_ids"}),flush=True)
        del loader,batch;gc.collect()
    assert loader_records[0]["ordered_clip_ids"] == loader_records[1]["ordered_clip_ids"]
    config=default_config();train_loader=make_loader(bench_train,config,training=True);valid_loader=make_loader(bench_valid,config,training=False)
    # Start worker processes outside measured steady model intervals.
    startup=time.perf_counter();warm_train=next(iter(train_loader));warm_valid=next(iter(valid_loader));startup=time.perf_counter()-startup
    worst=select_lengths(full_train,[16]*8)
    worst_cpu=collate_clips([worst[i] for i in range(8)])
    benchmarks=[]
    for name in ("model1","model2","model3"):
        c=default_config(name);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=build_model(c).to("cuda");policy=UnsupportedClassPolicy(model,counts)
        criterion=MultilabelLoss(c["loss"],policy.active_mask).to("cuda");optimizer=make_optimizer(model,c)
        worst_gpu=move_batch_to_device(worst_cpu,"cuda");model.train()
        z=model(worst_gpu["frames"],worst_gpu["lengths"],worst_gpu["valid_frame_mask"])
        criterion(z,worst_gpu["target"]).backward();optimizer.zero_grad(set_to_none=True)
        del worst_gpu,z
        train_loader.sampler.set_epoch(1);torch.cuda.synchronize();begin=time.perf_counter()
        trained=train_one_epoch(model,train_loader,optimizer,criterion,policy,"cuda",accumulation_steps=4,grad_clip_norm=5.)
        torch.cuda.synchronize();train_seconds=time.perf_counter()-begin;begin=time.perf_counter()
        evaluated=evaluate_loader(model,valid_loader,criterion,"cuda")
        torch.cuda.synchronize();valid_seconds=time.perf_counter()-begin
        assert trained["examples"] == 32 and evaluated["examples"] == 16 and trained["optimizer_updates"] == 1
        row=dict(model=name,Train_diagnostic_clips=32,Valid_diagnostic_clips=16,optimizer_updates=1,
                 train_seconds=train_seconds,valid_seconds=valid_seconds,workers=2,precision="fp32",worst_case_batch_shape=[8,16,3,224,398],
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 estimated_full_epoch_seconds=max(train_seconds/32,loader_records[1]["steady_seconds_per_clip"])*4667+valid_seconds/16*979,
                 budget_estimate_only_not_full_training=True)
        benchmarks.append(row);print(json.dumps(row),flush=True)
        del model,policy,criterion,optimizer,trained,evaluated;gc.collect();torch.cuda.empty_cache()
    del train_loader,valid_loader,warm_train,warm_valid;gc.collect()
    first_config=load_config(OUT/"configs/model1_seed42_bce.json")
    nominal=sum(r["estimated_full_epoch_seconds"] for r in benchmarks)*first_config["training"]["max_epochs"]
    plan=dict(first_round_models=["model1","model2","model3"],seed=42,loss="BCE",loss_active_classes=118,output_classes=119,
              unsupported_ids=[23],backbone_frozen=True,ImageNet_initialization="same verified ResNet18 IMAGENET1K_V1",microbatch=8,gradient_accumulation=4,effective_batch=32,
              Train_microbatches_per_epoch=math.ceil(4667/8),optimizer_updates_per_epoch=math.ceil(math.ceil(4667/8)/4),Valid_batches_per_epoch=math.ceil(979/8),
              max_epochs=first_config["training"]["max_epochs"],patience=first_config["training"]["patience"],optimizer="AdamW",peak_lr=.001,weight_decay=.0001,warmup_epochs=2,schedule="cosine to 0.1 peak",precision="FP32",num_workers_per_loader=2,
              model_selection="strict highest Valid fixed31 mAP; ties keep earlier checkpoint",threshold="best-checkpoint Valid fixed31 Macro-F1 grid0.05..0.95 step0.01; lowest tie",
              nominal_three_models_max_budget_wall_hours=nominal/3600,rough_range_hours=[nominal*.7/3600,nominal*2/3600],
              assumptions="32 Train/16 Valid small warm-worker benchmark; early stopping reduces cost; CPU decode, thermal throttling and OS caching add uncertainty",
              raw_epoch_estimates=benchmarks,actual_formal_training_started=False,Test_used=False,automatic_followup=False)
    write_json(OUT/"first_round_plan.json",plan)
    summary=dict(status="passed",unit_tests=result.testsRun,unit_failures=0,unit_errors=0,real_BCE_ASL_cases=diagnostic_records,
                 same_input_across_three_models_and_losses=True,loader_benchmarks=loader_records,worker_pools_startup_seconds=startup,
                 pipeline_benchmarks=benchmarks,diagnostic_optimizer_updates_total=9,diagnostic_checkpoints_are_not_formal_models=True,
                 benchmark_Train_clip_ids=[r["clip_id"] for r in bench_train.records],benchmark_Valid_clip_ids=[r["clip_id"] for r in bench_valid.records],
                 GPU=torch.cuda.get_device_name(0),GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                 total_seconds=time.perf_counter()-started,full_data_audit_passed=True,full_features_extracted=False,formal_training=False,Test_opened=False,next_stage_started=False)
    write_json(OUT/"preparation_validation.json",summary)
    print(json.dumps(dict(status="passed",unit_tests=result.testsRun,real_model_loss_cases=6,diagnostic_optimizer_updates_total=9,total_seconds=summary["total_seconds"])),flush=True)


if __name__ == "__main__":
    main()
