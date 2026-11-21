"""Shared HTTP client: model objects live only in the independent backend."""
import json
import os
from urllib.request import Request,urlopen
from urllib.parse import urlencode,quote,urlparse
from urllib.error import HTTPError,URLError
import streamlit as st
from threading import RLock
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
    @property
    def loaded_devices(self):
        return set(self._direct.predictors) if DIRECT and self._direct is not None else set() if DIRECT else set(request("/health")["loaded_devices"])
    def predict(self,path,device="cuda",top_k=119,model="attention",start=0.,end=None):
        if DIRECT:
            with self._initialization_lock:
                if self._direct is None:
                    from animalact2.dashboard.predictor import Runtime as BackendRuntime
                    self._direct=BackendRuntime()
            return self._direct.predict(path,key=model,device=device,start=start,end=end)
        q=dict(model=model,device=device,start=start)
        if end is not None: q["end"]=end
        return request("/api/predict?"+urlencode(q),path.read_bytes(),{"Content-Type":"application/octet-stream","X-Filename":quote(path.name)})

@st.cache_resource(show_spinner=False)
def get_runtime(): return Runtime()

@st.cache_data(show_spinner=False)
def model_info(key="attention"):
    if DIRECT:
        from animalact2.dashboard.registry import model_info as info
        return info(key)
    return next(r for r in request("/api/models") if r["model_id"]==key)

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
