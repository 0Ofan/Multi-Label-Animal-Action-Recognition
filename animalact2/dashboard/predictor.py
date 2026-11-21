"""Online frozen VideoMAE + exact Phase 2.4 Seed42 head. No feature cache writes."""
import time
import threading
import hashlib
from pathlib import Path
import torch
from deployment.encoder import FrozenVideoMAEAdapter
from animalact2.phase2_2.heads import VideoMAEProposalHead
from .registry import ROOT, model_info, classes, sha, read_json
from .video import prepare_video

class Predictor:
    def __init__(self, device="cuda"):
        if device not in ("cuda", "cpu"): raise ValueError("Device must be cuda or cpu")
        if device=="cuda" and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable")
        self.device=device
        torch.set_num_threads(4)
        # Match the FP32 cache extractor and formal training numerical policy.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        self.encoder=FrozenVideoMAEAdapter(device).eval()
        self.heads={}

    def load_head(self, key):
        if key not in self.heads:
            info=model_info(key)
            if sha(info["checkpoint"])!=info["checkpoint_SHA256"]: raise ValueError("Checkpoint SHA256 mismatch")
            ck=torch.load(info["checkpoint"],map_location="cpu",weights_only=True)
            if ck["epoch"]!=info["best_epoch"] or ck["config_SHA256"]!=info["config_SHA256"] or ck["schema"]!="animalact2-head-v1":
                raise ValueError("Portable checkpoint metadata mismatch")
            if ck["source_checkpoint_SHA256"]!=info["source_checkpoint_SHA256"]:
                raise ValueError("Original training checkpoint mismatch")
            head=VideoMAEProposalHead(info["head"])
            head.load_state_dict(ck["head_state"],strict=True)
            self.heads[key]=head.requires_grad_(False).eval().to(self.device)
        return self.heads[key]

    @torch.inference_mode()
    def predict(self,path,key="attention",start=0.,end=None):
        begin=time.perf_counter(); head=self.load_head(key); info=model_info(key)
        pixels,meta=prepare_video(path,start,end); ready=time.perf_counter()
        encoded=self.encoder(pixels.to(self.device))
        features=encoded["features"]
        mask=torch.ones(features.shape[:2],dtype=torch.bool,device=self.device)
        logits=head(features,torch.tensor([features.shape[1]]),mask)
        scores=logits.sigmoid()[0].cpu()
        rows=[]; threshold32=torch.tensor(info["threshold"],dtype=torch.float32).item()
        for row,p in zip(classes(),scores.tolist()):
            eligible=row["class_id"]!=23
            rows.append(dict(**row,score=p,above_threshold=p>=threshold32,eligible_for_prediction=eligible,
                             selected=eligible and p>=threshold32,learning_status="Train_supported" if eligible else "No_Train_positive"))
        ranked=sorted(rows,key=lambda r:(-r["score"],r["class_id"]))
        elapsed=time.perf_counter()-begin
        return dict(schema_version=2,video_id=Path(path).stem,model_id=key,model_name=info["model_name"],
            seed=42,best_epoch=info["best_epoch"],threshold=info["threshold"],checkpoint_SHA256=info["checkpoint_SHA256"],
            device=self.device,logits=logits[0].cpu().tolist(),class_scores=rows,
            probabilities={r["name"]:r["score"] for r in rows},predicted_labels=[r["name"] for r in ranked if r["selected"]],
            top_k=ranked,inference_time=elapsed,video=meta,timings=dict(prepare_seconds=ready-begin,encoder_head_seconds=elapsed-(ready-begin)),
            feature_shape=list(features.shape),videomae_geometry=encoded["geometry"],unsupported_class_ids=[23],evaluation_unit="supplied_clip",calibrated=False)

class Runtime:
    def __init__(self):
        self.predictors={}; self.lock=threading.RLock()
    def predict(self,path,key="attention",device="cuda",start=0.,end=None):
        with self.lock:
            started=time.perf_counter()
            if device not in self.predictors: self.predictors[device]=Predictor(device)
            result=self.predictors[device].predict(path,key,start,end)
            result["request_seconds"]=time.perf_counter()-started
            return result
