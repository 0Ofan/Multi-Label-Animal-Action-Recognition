from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
from animalact2.phase1_5_cache_pilot.support import sha,digest,RAMMonitor,forbidden

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PILOT=ROOT/"animalact2/phase1_5_cache_pilot"


def guard(event,args):
    if event in ("socket.connect","socket.getaddrinfo"):raise RuntimeError("No network in full cache generation")
    if event!="open" or not isinstance(args[0],(str,bytes)):return
    p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
    write=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
    if write and not p.is_relative_to(OUT):raise RuntimeError("Write outside independent full cache: "+str(p))
    if p.is_relative_to(ROOT) and not p.is_relative_to(OUT):
        if forbidden(p):raise RuntimeError("Test/predictions/Model6 input blocked")
        if p.suffix.lower() in (".pt",".pth",".npy",".npz",".safetensors"):
            if p!=ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth" and not (p.is_relative_to(PILOT) and p.suffix.lower()==".npy"):
                raise RuntimeError("Old checkpoint/feature input blocked")


def enable_guard():sys.addaudithook(guard)


def write_json(path,obj):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT):raise ValueError("Full-cache-local output required")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def snapshot():
    result={};skip={".venv",".git",".codex",".agents",".aws","node_modules","__pycache__"}
    for directory,sub,files in os.walk(ROOT):
        p=Path(directory);sub[:]=[n for n in sub if n not in skip and "model6" not in n.lower() and (p/n).resolve()!=OUT]
        for name in files:
            path=p/name
            if "model6" in name.lower() or path.is_symlink():continue
            st=path.stat();result[path.relative_to(ROOT).as_posix()]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return result


def hashes():
    result={}
    for d in (ROOT/"animalact2").iterdir():
        if not d.is_dir() or d.resolve()==OUT:continue
        for p in d.rglob("*"):
            if not p.is_file() or forbidden(p):continue
            tensor=p.suffix.lower() in (".pt",".pth",".npy",".npz",".safetensors")
            if tensor and not (p.is_relative_to(PILOT) and p.suffix.lower()==".npy"):continue
            result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def capture():
    p=OUT/"protection_baseline.json"
    if p.exists():raise FileExistsError("Keep initial historical baseline")
    obj=dict(metadata=snapshot(),historical_SHA256=hashes(),official_weights_SHA256=sha(ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth"),started_unix=time.time())
    write_json(p,obj);print(json.dumps(dict(metadata_files=len(obj["metadata"]),historical_hashed_files=len(obj["historical_SHA256"]))))


def verify():
    obj=json.loads((OUT/"protection_baseline.json").read_text(encoding="utf-8"))
    assert snapshot()==obj["metadata"],"Historical file metadata changed"
    assert hashes()==obj["historical_SHA256"],"Historical/Pilot file contents changed"
    assert sha(ROOT/".cache/torch/checkpoints/resnet18-f37072fd.pth")==obj["official_weights_SHA256"]
    result=dict(status="passed",historical_metadata_files=len(obj["metadata"]),historical_hashed_files=len(obj["historical_SHA256"]),
                Pilot_cache_and_history_unchanged=True,release_models_configs_checkpoints_unchanged=True,Test_content_opened=False)
    write_json(OUT/"protection_verification.json",result);return result


if __name__=="__main__":
    enable_guard()
    if sys.argv[1:]==["capture"]:capture()
    elif sys.argv[1:]==["verify"]:print(json.dumps(verify(),indent=2))
