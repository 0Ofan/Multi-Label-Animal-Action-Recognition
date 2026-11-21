from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
from animalact2.phase1_5_cache_pilot.support import sha,digest,RAMMonitor,forbidden

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
FULL=ROOT/"animalact2/phase1_5_cache_full"
PILOT=ROOT/"animalact2/phase1_5_cache_pilot"


def guard(event,args):
    if event in ("socket.connect","socket.getaddrinfo"):raise RuntimeError("No network during this formal round")
    if event!="open" or not isinstance(args[0],(str,bytes)):return
    p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
    write=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
    if write and not p.is_relative_to(OUT):raise RuntimeError("Only first_round outputs may be written: "+str(p))
    if p.is_relative_to(ROOT) and not p.is_relative_to(OUT):
        low="/"+p.as_posix().lower()
        if forbidden(p) or "/videos/" in low or "/annotations/" in low:raise RuntimeError("Raw video/annotation/Test/prediction source blocked")
        if p.suffix.lower() in (".pt",".pth",".npy",".npz",".safetensors"):
            official=p==ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth"
            allowed_array=p.suffix.lower()==".npy" and (p.is_relative_to(FULL) or p.is_relative_to(PILOT))
            if not official and not allowed_array:raise RuntimeError("Old task checkpoint or unrelated features blocked")


def enable_guard():
    # Process-local runtime files (including spawned workers), never Windows/global env settings.
    import tempfile
    runtime=OUT/".runtime_temp";runtime.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(runtime)
    os.environ["TEMP"]=str(runtime);os.environ["TMP"]=str(runtime)
    os.environ["TORCHINDUCTOR_CACHE_DIR"]=str(runtime/"torchinductor")
    sys.addaudithook(guard)


def write_json(path,obj):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT):raise ValueError("Output outside first_round")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def owned(path):
    p=Path(path).resolve()
    if not p.is_relative_to(OUT):raise ValueError("Only first_round checkpoints/outputs")
    return p


def snapshot():
    result={};skip={".venv",".git",".codex",".agents",".aws","node_modules","__pycache__"}
    for directory,sub,files in os.walk(ROOT):
        p=Path(directory);sub[:]=[n for n in sub if n not in skip and "model6" not in n.lower() and (p/n).resolve()!=OUT]
        for name in files:
            path=p/name
            if "model6" in name.lower() or path.is_symlink():continue
            st=path.stat();result[path.relative_to(ROOT).as_posix()]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return result


def historical_hashes():
    result={}
    for d in (ROOT/"animalact2").iterdir():
        if not d.is_dir() or d.resolve()==OUT:continue
        for p in d.rglob("*"):
            if not p.is_file() or forbidden(p):continue
            tensor=p.suffix.lower() in (".pt",".pth",".npy",".npz",".safetensors")
            if tensor and not (p.suffix.lower()==".npy" and (p.is_relative_to(FULL) or p.is_relative_to(PILOT))):continue
            result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def capture():
    path=OUT/"protection_baseline.json"
    if path.exists():return
    obj=dict(metadata=snapshot(),historical_SHA256=historical_hashes(),official_weights_SHA256=sha(ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth"),started_unix=time.time())
    write_json(path,obj);print(json.dumps(dict(protection="captured",metadata_files=len(obj["metadata"]),hashed_files=len(obj["historical_SHA256"]))))


def verify():
    obj=json.loads((OUT/"protection_baseline.json").read_text(encoding="utf-8"))
    assert snapshot()==obj["metadata"],"Historical metadata changed"
    assert historical_hashes()==obj["historical_SHA256"],"History/Full Cache/Pilot content changed"
    assert sha(ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth")==obj["official_weights_SHA256"]
    result=dict(status="passed",historical_metadata_files=len(obj["metadata"]),historical_hashed_files=len(obj["historical_SHA256"]),
                release_Full_Cache_Pilot_and_historical_models_unchanged=True,Test_read=False)
    write_json(OUT/"protection_verification.json",result);return result


def worker_init(worker_id):
    import torch
    enable_guard();torch.set_num_threads(1)


if __name__=="__main__":
    enable_guard()
    if sys.argv[1:]==["capture"]:capture()
    elif sys.argv[1:]==["verify"]:print(json.dumps(verify(),indent=2))
