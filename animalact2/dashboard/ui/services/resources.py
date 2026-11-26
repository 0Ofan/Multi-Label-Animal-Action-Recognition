"""Shared HTTP client: model objects live only in the independent backend."""
import json
import os
import time
from urllib.request import Request,urlopen
from urllib.parse import urlencode,quote,urlparse
from urllib.error import HTTPError,URLError
import streamlit as st
from threading import RLock
from pathlib import Path
from deployment.final_artifacts import declared
RESNET_MODELS = {"resnet_" + kind: kind for kind in ("mean", "bigru_mean", "bigru_attention")}
MODEL_OPTIONS = ["resnet_ensemble"] + list(RESNET_MODELS) + ["bigru_mean", "attention"]

def detection_checkpoint():
    root = Path(__file__).resolve().parents[4] / "proposal/action_detection_balanced_results"
    if not ((root / "detector.pt").exists() or declared(root / "detector.pt")) or not (root / "results.json").exists():
        return None
    report = json.loads((root / "results.json").read_text(encoding="utf-8"))
    return root / "detector.pt" if report.get("completed") and report.get("enabled_for_inference") and not report.get("smoke", True) and report.get("schema") == "automatic-actions-v1" else None

def available_models():
    root = Path(__file__).resolve().parents[4] / "proposal/results"
    return [key for key in MODEL_OPTIONS if key not in RESNET_MODELS or
            ((root / "results.json").exists() and ((root / (RESNET_MODELS[key] + ".pt")).exists()
             or declared(root / (RESNET_MODELS[key] + ".pt"))))]

def model_label(key):
    if key == "resnet_ensemble":
        return "ResNet-18 · Weighted ensemble"
    if key in RESNET_MODELS:
        return "ResNet-18 · " + RESNET_MODELS[key]
    return "VideoMAE · " + ("bigru_attention" if key == "attention" else key)

def whole_video_results():
    path = Path(__file__).resolve().parents[4] / "proposal/results/results.json"
    return json.loads(path.read_text(encoding="utf-8"))
DIRECT=os.environ.get("ANIMALACT2_BACKEND","direct")=="direct"
BASE=os.environ.get("ANIMALACT2_API_URL","http://127.0.0.1:8765")
if urlparse(BASE).hostname not in ("127.0.0.1","localhost"): raise ValueError("Local backend required")

def request(path,data=None,headers=None):
    try:
        with urlopen(Request(BASE+path,data=data,headers=headers or {}),timeout=300) as r: return json.load(r)
    except HTTPError as exc:
        try: message=json.load(exc).get("error",str(exc))
        except Exception: message=str(exc)
        raise RuntimeError(message) from exc
    except URLError as exc: raise RuntimeError("Start the local API on port 8765: "+str(exc.reason)) from exc

class Runtime:
    def __init__(self):
        self._direct=None
        self._initialization_lock=RLock()
        self._resnet={}
        self._ensemble=None
    @property
    def loaded_devices(self):
        devices = set(self._direct.predictors) if DIRECT and self._direct is not None else set() if DIRECT else set(request("/health")["loaded_devices"])
        return devices | set(self._resnet) | ({"cpu"} if self._ensemble is not None else set())
    def _with_actions(self, path, result, device, start, end):
        checkpoint = detection_checkpoint()
        if checkpoint is None:
            result["temporal_detection"] = dict(status="unavailable", segments=[])
        elif start != 0. or end is not None:
            result["temporal_detection"] = dict(status="unsupported_window", segments=[])
        else:
            started = time.perf_counter()
            try:
                with self._initialization_lock:
                    if device not in self._resnet:
                        from proposal.predict import ProposalPredictor
                        self._resnet[device] = ProposalPredictor(device)
                    from proposal.action_detection import predict_actions
                    result["temporal_detection"] = predict_actions(self._resnet[device], path, checkpoint)
            except Exception as exc:
                result["temporal_detection"] = dict(status="unavailable", segments=[], error=str(exc))
            result["classification_inference_time"] = result["inference_time"]
            result["temporal_detection"]["inference_time"] = time.perf_counter() - started
            result["inference_time"] += result["temporal_detection"]["inference_time"]
        return result
    def predict(self,path,device="cuda",top_k=119,model="attention",start=0.,end=None,query=None):
        if model == "resnet_ensemble":
            if start != 0. or end is not None or device != 'cpu':
                raise ValueError('Weighted ensemble uses complete videos on CPU.')
            with self._initialization_lock:
                if self._ensemble is None:
                    from ensemble_integration.adapter import IntegratedEngine
                    self._ensemble=IntegratedEngine()
                result=self._ensemble.predict(path)
            return self._with_actions(path,result,device,start,end)
        if model in RESNET_MODELS:
            if start != 0. or end is not None:
                raise ValueError("Whole-video models require the complete video.")
            with self._initialization_lock:
                if device not in self._resnet:
                    from proposal.predict import ProposalPredictor
                    self._resnet[device] = ProposalPredictor(device)
                result = self._resnet[device].predict(path, RESNET_MODELS[model], query)
            return self._with_actions(path, result, device, start, end)
        if DIRECT:
            with self._initialization_lock:
                if self._direct is None:
                    from animalact2.dashboard.predictor import Runtime as BackendRuntime
                    self._direct=BackendRuntime()
            result = self._direct.predict(path,key=model,device=device,start=start,end=end)
            return self._with_actions(path, result, device, start, end)
        q=dict(model=model,device=device,start=start)
        if end is not None: q["end"]=end
        result = request("/api/predict?"+urlencode(q),path.read_bytes(),{"Content-Type":"application/octet-stream","X-Filename":quote(path.name)})
        return self._with_actions(path, result, device, start, end)

@st.cache_resource(show_spinner=False)
def get_runtime(): return Runtime()

@st.cache_data(show_spinner=False)
def model_info(key="attention"):
    if key == "resnet_ensemble":
        return dict(head="BiGRU Mean 20% + Attention 80%",backbone="ResNet-18",seed=6116,
                    best_epoch="39 / 39",threshold=.20,evaluation_unit="whole_video")
    if key in RESNET_MODELS:
        result = whole_video_results()
        item = result["models"][RESNET_MODELS[key]]
        return dict(head=RESNET_MODELS[key], backbone="ResNet-18", seed=result["seed"],
                    best_epoch=item["selected_epoch"], threshold=item["threshold"],
                    evaluation_unit="whole_video")
    if DIRECT:
        from animalact2.dashboard.registry import model_info as info
        return dict(info(key), backbone="VideoMAE", evaluation_unit="supplied_clip")
    return dict(next(r for r in request("/api/models") if r["model_id"]==key), backbone="VideoMAE", evaluation_unit="supplied_clip")

@st.cache_data(show_spinner=False)
def cuda_available():
    import torch
    return torch.cuda.is_available()

@st.cache_data(show_spinner=False)
def evaluation(kind="summary",model="model3",seed=42):
    if DIRECT:
        from animalact2.dashboard.evaluation_data import evaluation as read
        return read(kind,model,seed)
    return request("/api/evaluation?"+urlencode(dict(kind=kind,model=model,seed=seed)))
