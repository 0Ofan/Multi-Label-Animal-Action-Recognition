"""VideoMAE cached training using the unchanged shared epoch/loss/evaluation math."""
import gc
import json
import time
import torch
from animalact2.phase1_5.engine import train_one_epoch,evaluate_loader,summarize,make_optimizer
from animalact2.phase1_5.data import training_class_support
from animalact2.phase1_5.losses import MultilabelLoss,UnsupportedClassPolicy
from animalact2.phase1_5.schedule import LearningRatePlan,EarlyStopping
from animalact2.phase1_5.config import config_hash,cohort_hash,development_fingerprints
from animalact2.phase1_5.evaluation import select_validation_threshold
from .models import build_model
from .data import CachedFeatureDataset,make_feature_loader
from .support import ROOT,OUT,write_json,owned,sha,digest,RAMMonitor
from . import checkpoint
from .exports import save_history,save_metrics,curves


class FullCacheTrainer:
    def __init__(self,config,output_dir,*,train_dataset=None,valid_dataset=None,diagnostic=False,counts=None,workers_override=None):
        self.config=config;self.output_dir=owned(output_dir);self.diagnostic=diagnostic
        if config["loss"]["name"]!="bce":raise ValueError("Only approved BCE is authorized")
        self.train_dataset=train_dataset if train_dataset is not None else CachedFeatureDataset("train")
        self.valid_dataset=valid_dataset if valid_dataset is not None else CachedFeatureDataset("valid")
        assert (self.train_dataset.split,self.valid_dataset.split)==("train","valid")
        if not diagnostic:
            assert (len(self.train_dataset),len(self.valid_dataset))==(4667,979)
            assert counts is None and workers_override is None
        self.device=torch.device(config["device"]);assert self.device.type=="cuda"
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        self.model=build_model(config).to(self.device);self.view=self.model;self.encoder_calls=0
        def forbid_encoder(module,args):
            self.encoder_calls+=1;raise RuntimeError("No visual encoder forward during cached training")
        self.encoder_hook=self.model.backbone.encoder.register_forward_pre_hook(forbid_encoder)
        train_counts=training_class_support(self.train_dataset) if counts is None else counts
        self.policy=UnsupportedClassPolicy(self.model,train_counts);assert self.policy.unsupported.tolist()==[23]
        self.criterion=MultilabelLoss(config["loss"],self.policy.active_mask).to(self.device)
        self.optimizer=make_optimizer(self.model,config)
        self.schedule=LearningRatePlan(self.optimizer,config)
        self.early=EarlyStopping(config["training"]["patience"],config["training"]["min_delta"])
        self.train_loader=make_feature_loader(self.train_dataset,config,training=True,workers_override=workers_override)
        self.valid_loader=make_feature_loader(self.valid_dataset,config,training=False,workers_override=workers_override)

    def make_state(self,epoch,history,resources):
        return checkpoint.create(self.model,self.optimizer,self.schedule,self.early,self.policy,self.config,epoch=epoch,history=history,diagnostic=self.diagnostic,resource_state=resources)

    def close(self):
        self.encoder_hook.remove();self.train_loader=None;self.valid_loader=None;self.view=None
        self.model.cpu();gc.collect();torch.cuda.empty_cache()

    def run(self,*,resume=None,execute_training=False):
        if self.diagnostic or not execute_training:raise RuntimeError("Formal execution flag and non-diagnostic trainer required")
        gate=json.loads((OUT/"integration_validation.json").read_text(encoding="utf-8"))
        if gate["status"]!="passed" or gate["integration_fingerprints"]!=checkpoint.fingerprints():raise RuntimeError("Current integration must pass validation before training")
        if gate["cache_provenance"]!=checkpoint.cache_provenance():raise RuntimeError("Validated Full Cache changed")
        prior=dict(wall_seconds=0.,CPU_RSS_parent_peak_bytes=0,CPU_RSS_parent_workers_sum_peak_bytes=0,GPU_allocated_peak_bytes=0,GPU_reserved_peak_bytes=0,resume_count=0)
        history=[];start_epoch=1
        if resume is None:
            if self.output_dir.exists():raise FileExistsError("Existing formal experiment must be resumed, not overwritten")
            self.output_dir.mkdir(parents=True)
        else:
            if owned(resume).parent!=self.output_dir:raise ValueError("Resume path belongs to another run")
            state=checkpoint.load(resume,self.model,self.config,optimizer=self.optimizer,schedule=self.schedule,early=self.early,policy=self.policy,restore_random=True)
            history=state["history"];start_epoch=state["epoch"]+1;prior=state["resources"];prior["resume_count"]+=1
            if self.early.best_epoch==state["epoch"]:checkpoint.save(self.output_dir/"best.pt",state)
            save_history(self.output_dir,history)
        write_json(self.output_dir/"config.json",self.config)
        write_json(self.output_dir/"provenance.json",dict(input_mode="videomae_full_cache_fp32",config_SHA256=config_hash(self.config),shared_cohort_SHA256=cohort_hash(self.config),
                   data_fingerprints=development_fingerprints(),cache_provenance=checkpoint.cache_provenance(),integration_fingerprints=checkpoint.fingerprints(),
                   class_policy=self.policy.metadata(),encoder_in_epoch_forbidden=True,Test_used=False))
        started=time.perf_counter()
        with RAMMonitor() as ram:
            def resources():
                memory=ram.result()
                return dict(wall_seconds=prior["wall_seconds"]+time.perf_counter()-started,
                            CPU_RSS_parent_peak_bytes=max(prior["CPU_RSS_parent_peak_bytes"],memory["parent_RSS_peak_bytes"]),
                            CPU_RSS_parent_workers_sum_peak_bytes=max(prior["CPU_RSS_parent_workers_sum_peak_bytes"],memory["parent_plus_workers_RSS_sum_peak_bytes"]),
                            GPU_allocated_peak_bytes=max(prior["GPU_allocated_peak_bytes"],torch.cuda.max_memory_allocated()),
                            GPU_reserved_peak_bytes=max(prior["GPU_reserved_peak_bytes"],torch.cuda.max_memory_reserved()),resume_count=prior["resume_count"],
                            RAM_scope=memory["scope"])
            for epoch in range(start_epoch,51):
                if self.early.bad_epochs>=10:break
                begin=time.perf_counter();self.train_loader.sampler.set_epoch(epoch);lr=self.schedule.apply_epoch(epoch)
                t=time.perf_counter()
                training=train_one_epoch(self.view,self.train_loader,self.optimizer,self.criterion,self.policy,self.device,accumulation_steps=4,grad_clip_norm=5.)
                torch.cuda.synchronize();train_seconds=time.perf_counter()-t;t=time.perf_counter()
                valid=evaluate_loader(self.view,self.valid_loader,self.criterion,self.device)
                torch.cuda.synchronize();valid_seconds=time.perf_counter()-t;t=time.perf_counter()
                assert training["examples"]==4667 and valid["examples"]==979 and training["optimizer_updates"]==146
                assert valid["clip_ids"]==[r["clip_id"] for r in self.valid_dataset.records]
                tm,vm=summarize(training,self.policy),summarize(valid,self.policy)
                metric_seconds=time.perf_counter()-t;improved,stop=self.early.update(vm["main_mAP"],epoch)
                record=dict(epoch=epoch,lr=lr,train_online_objective=training["objective"],valid_objective=valid["objective"],
                            train_online_mAP=tm["main_mAP"],valid_mAP=vm["main_mAP"],train_online_Macro_F1_at_05=tm["main_macro_f1"],valid_Macro_F1_at_05=vm["main_macro_f1"],
                            train_online_BCE119=tm["BCE119"],valid_BCE119=vm["BCE119"],train_online_BCE118=tm["BCE_Train_supported118"],valid_BCE118=vm["BCE_Train_supported118"],
                            optimizer_updates=146,train_seconds=train_seconds,valid_seconds=valid_seconds,metric_seconds=metric_seconds,
                            epoch_seconds=time.perf_counter()-begin,bad_epochs=self.early.bad_epochs,best_epoch=self.early.best_epoch,
                            Train_input_order_SHA256=digest(training["clip_ids"]),Train_targets_SHA256=digest(training["target"].tolist()),encoder_forward_calls=self.encoder_calls)
                history.append(record);state=self.make_state(epoch,history,resources())
                checkpoint.save(self.output_dir/"last.pt",state)
                if improved:checkpoint.save(self.output_dir/"best.pt",state)
                save_history(self.output_dir,history)
                print(json.dumps(dict(model=self.config["model"],**record),allow_nan=False),flush=True)
                del state,training,valid,tm,vm
                if stop:break
            assert history and self.encoder_calls==0
            best=checkpoint.load(self.output_dir/"best.pt",self.model,self.config,policy=self.policy)
            best_valid=evaluate_loader(self.view,self.valid_loader,self.criterion,self.device)
            threshold=select_validation_threshold(best_valid["logits"],best_valid["target"],split="valid")
            metrics=summarize(best_valid,self.policy,threshold=threshold["threshold"])
            assert abs(metrics["main_mAP"]-max(r["valid_mAP"] for r in history))<1e-12
            save_metrics(self.output_dir/"best_valid",metrics)
            write_json(self.output_dir/"threshold_selection.json",threshold)
            torch.save(dict(logits=best_valid["logits"],target=best_valid["target"],clip_ids=best_valid["clip_ids"],parent_video_ids=best_valid["parent_video_ids"],
                            threshold=threshold["threshold"],class_policy=self.policy.metadata(),cache_provenance=checkpoint.cache_provenance(),
                            data_fingerprints=development_fingerprints(),checkpoint_SHA256=sha(self.output_dir/"best.pt")),self.output_dir/"valid_predictions.pt")
            curves(self.output_dir,history)
            final_resources=resources()
            results=dict(status="completed",model=self.config["model"],config=self.config,config_SHA256=config_hash(self.config),shared_cohort_SHA256=cohort_hash(self.config),
                         best_epoch=best["epoch"],completed_epochs=len(history),stop_reason="early_stopping_patience10" if self.early.bad_epochs>=10 else "max_epochs50",
                         valid=metrics,threshold_selection=threshold,resources=final_resources,training_epoch_seconds=sum(r["epoch_seconds"] for r in history),
                         per_epoch_time_scope="Train+Valid+metrics, excludes checkpoint/log writes; resources.wall_seconds includes them and final Valid exports",
                         peak_memory_scope="PyTorch allocation/reservation and sampled Windows parent+own workers RSS; no driver/desktop GPU memory, shared pages may double count",
                         checkpoint="best.pt",checkpoint_SHA256=sha(self.output_dir/"best.pt"),last_checkpoint="last.pt",cache_provenance=checkpoint.cache_provenance(),
                         encoder_forward_calls=self.encoder_calls,Test_used=False,ASL_or_additional_experiments=False)
            write_json(self.output_dir/"results.json",results)
            print(json.dumps(dict(model=self.config["model"],completed=True,best_epoch=best["epoch"],epochs=len(history),valid_mAP=metrics["main_mAP"],valid_Macro_F1=metrics["main_macro_f1"],BCE119=metrics["BCE119"])),flush=True)
        return results
