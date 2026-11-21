from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from animalact2.phase2_2.support import sha,digest,RAMMonitor
from animalact2.phase2_3_full_cache.support import protected_hashes as earlier_hashes

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
FULL=ROOT/"animalact2/phase2_3_full_cache"
PILOT=ROOT/"animalact2/phase2_2"


def owned(path):
    p=Path(path).resolve()
    if not p.is_relative_to(OUT):raise ValueError("Only phase2_4 outputs")
    return p


def write_json(path,obj):
    p=owned(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def enable_guard():
    runtime=OUT/".runtime";runtime.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(runtime)
    os.environ.update(TEMP=str(runtime),TMP=str(runtime),TORCHINDUCTOR_CACHE_DIR=str(runtime/"torchinductor"),HF_HOME=str(runtime/"hf"),
                      HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
    def guard(event,args):
        if event in ("socket.connect","socket.getaddrinfo"):raise RuntimeError("No network during cached training")
        if event!="open" or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        write=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if write and not p.is_relative_to(OUT):raise RuntimeError("Write outside phase2_4: "+str(p))
        if p.is_relative_to(ROOT) and not p.is_relative_to(OUT):
            low="/"+p.as_posix().lower()
            if any(s in low for s in ("/videos/","/annotations/","/test_metadata_only/","predictions","model6")):
                raise RuntimeError("Raw video/annotation/Test/old predictions blocked")
            if p.suffix.lower() in (".pt",".pth",".npy",".npz",".safetensors") and not (p.suffix.lower()==".npy" and p.is_relative_to(FULL/"cache")):
                raise RuntimeError("Only full VideoMAE arrays may be loaded; no encoder or old task weights")
    sys.addaudithook(guard)


def worker_init(worker_id):
    import torch
    enable_guard();torch.set_num_threads(1)


def snapshot():
    result={};skip={".venv",".git",".codex",".agents",".aws","__pycache__","node_modules"}
    for directory,sub,files in os.walk(ROOT):
        d=Path(directory);sub[:]=[n for n in sub if n not in skip and "model6" not in n.lower() and (d/n).resolve()!=OUT]
        for name in files:
            p=d/name
            if p.is_symlink() or "model6" in name.lower():continue
            st=p.stat();result[p.relative_to(ROOT).as_posix()]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return result


def protected_hashes():
    result=earlier_hashes()
    manifest=json.loads((FULL/"artifact_manifest.json").read_text(encoding="utf-8"))
    for name,entry in manifest["files"].items():
        p=FULL/name;actual=sha(p)
        if actual!=entry["sha256"]:raise ValueError("Full-cache artifact changed: "+name)
        result[p.relative_to(ROOT).as_posix()]=actual
    for name in ("artifact_manifest.json","FINAL_VERIFICATION.json"):
        p=FULL/name;result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def capture():
    p=OUT/"protection_baseline.json"
    if p.exists():return
    write_json(p,dict(started_unix=time.time(),metadata=snapshot(),SHA256=protected_hashes()))


def verify():
    baseline=json.loads((OUT/"protection_baseline.json").read_text(encoding="utf-8"))
    assert snapshot()==baseline["metadata"],"Historical metadata changed"
    assert protected_hashes()==baseline["SHA256"],"Protected content changed"
    result=dict(status="passed",historical_metadata_files=len(baseline["metadata"]),historical_hashed_files=len(baseline["SHA256"]),
                release_full_cache_Pilot_ResNet_and_weights_unchanged=True,Test_content_read=False)
    write_json(OUT/"protection_verification.json",result);return result


if __name__=="__main__":
    # Old checkpoints are byte-hashed for preservation only, never deserialized.
    if sys.argv[1:]==["verify"]:print(json.dumps(verify()),flush=True)
    else:capture();print("phase2_4 history/full-cache integrity baseline captured",flush=True)
