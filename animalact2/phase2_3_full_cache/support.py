from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from animalact2.phase2_2.support import sha,digest,protected_hashes as previous_hashes,RAMMonitor

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PILOT=ROOT/"animalact2/phase2_2"
RELEASE=ROOT/"animalact2/release_v1"


def write_json(path,value):
    p=Path(path).resolve()
    if not p.is_relative_to(OUT):raise ValueError("Full VideoMAE output only")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def enable_guard():
    runtime=OUT/".runtime";runtime.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(runtime)
    for key in ("TEMP","TMP","TORCHINDUCTOR_CACHE_DIR","MPLCONFIGDIR"):
        os.environ[key]=str(runtime/key.lower())
        if key in ("TEMP","TMP"):Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(runtime/"hf"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
    source=json.loads((ROOT/".cache/videomae_source.json").read_text(encoding="utf-8"))
    weight=(Path(source["path"])/"model.safetensors").resolve()
    def guard(event,args):
        if event in ("socket.connect","socket.getaddrinfo"):raise RuntimeError("No network in full-cache extraction")
        if event!="open" or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and not p.is_relative_to(OUT):raise RuntimeError("Only phase2_3_full_cache may be written: "+str(p))
        if p.is_relative_to(ROOT) and not p.is_relative_to(OUT):
            low="/"+p.as_posix().lower()
            if any(s in low for s in ("/videos/test/","/annotations/test/","/test_metadata_only/","predictions","model6")):
                raise RuntimeError("Test/predictions/excluded models are blocked")
            if p.suffix.lower() in (".pt",".pth",".npy",".npz",".safetensors"):
                if p!=weight and not (p.suffix.lower()==".npy" and p.is_relative_to(PILOT/"cache")):
                    raise RuntimeError("Only original VideoMAE and validated Pilot arrays are readable")
    sys.addaudithook(guard)


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
    result=previous_hashes()
    inventory=json.loads((PILOT/"FINAL_VERIFICATION.json").read_text(encoding="utf-8"))["artifacts"]
    for name,entry in inventory.items():
        p=PILOT/name;actual=sha(p)
        if actual!=entry["sha256"]:raise ValueError("Pilot integrity failure: "+name)
        result[p.relative_to(ROOT).as_posix()]=actual
    result[(PILOT/"FINAL_VERIFICATION.json").relative_to(ROOT).as_posix()]=sha(PILOT/"FINAL_VERIFICATION.json")
    return result


def capture():
    p=OUT/"protection_baseline.json"
    if p.exists():return
    write_json(p,dict(started_unix=time.time(),metadata=snapshot(),SHA256=protected_hashes()))


def verify():
    old=json.loads((OUT/"protection_baseline.json").read_text(encoding="utf-8"))
    assert snapshot()==old["metadata"],"Historical metadata changed"
    assert protected_hashes()==old["SHA256"],"Protected content changed"
    r=dict(status="passed",historical_metadata_files=len(old["metadata"]),historical_content_hashed_files=len(old["SHA256"]),
           Pilot_release_pretrained_and_ResNet_experiments_unchanged=True,Test_content_read=False)
    write_json(OUT/"protection_verification.json",r);return r


if __name__=="__main__":
    # Integrity-only reads of old checkpoints are raw byte hashing, not model loading.
    if sys.argv[1:]==["verify"]:print(json.dumps(verify()),flush=True)
    else:capture();print("Full VideoMAE history/Pilot baseline captured",flush=True)
