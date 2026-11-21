from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from animalact2.phase1_5_cache_pilot.support import RAMMonitor

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RELEASE=ROOT/"animalact2/release_v1"
SOURCE=ROOT/".cache/videomae_source.json"


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    p=Path(path).resolve()
    if not p.is_relative_to(OUT):raise ValueError("Pilot output only")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def enable_guard():
    runtime=OUT/".runtime";runtime.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(runtime)
    for key in ("TEMP","TMP","TORCHINDUCTOR_CACHE_DIR","MPLCONFIGDIR"):os.environ[key]=str(runtime/key.lower())
    # Process-local settings only; runtime defaults cannot write the original model cache.
    for key in ("TEMP","TMP"):Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(runtime/"hf"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
    source=json.loads(SOURCE.read_text(encoding="utf-8"));weights=Path(source["path"]).resolve()/"model.safetensors"
    def guard(event,args):
        if event in ("socket.connect","socket.getaddrinfo"):raise RuntimeError("No network in VideoMAE pilot")
        if event!="open" or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and not p.is_relative_to(OUT):raise RuntimeError("Write outside phase2_2: "+str(p))
        if p.is_relative_to(ROOT) and not p.is_relative_to(OUT):
            low="/"+p.as_posix().lower()
            if any(x in low for x in ("/videos/test/","/annotations/test/","/test_metadata_only/","predictions","model6")):
                raise RuntimeError("Test, old predictions and excluded models are blocked")
            if p.suffix.lower() in (".pt",".pth",".npy",".npz",".safetensors") and p!=weights:
                raise RuntimeError("Only the original VideoMAE pretrained weights may be loaded")
    sys.addaudithook(guard)


def metadata_snapshot():
    result={};skip={".venv",".git",".codex",".agents",".aws","__pycache__","node_modules"}
    for directory,sub,files in os.walk(ROOT):
        d=Path(directory);sub[:]=[s for s in sub if s not in skip and "model6" not in s.lower() and (d/s).resolve()!=OUT]
        for name in files:
            p=d/name
            if p.is_symlink() or "model6" in name.lower():continue
            st=p.stat();result[p.relative_to(ROOT).as_posix()]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return result


def protected_hashes():
    # No Test label/prediction source is opened, including for content hashing.
    paths=[]
    for sub in ("train","valid"):
        paths.extend(p for p in (RELEASE/sub).rglob("*") if p.is_file())
    paths.extend(RELEASE/n for n in ("version.json","class_mapping.json","main_evaluation_classes.json","preprocessing.json","development_index.json","SHA256SUMS.txt"))
    for d in (ROOT/"animalact2/phase1_5_1",ROOT/"animalact2/phase1_5"):
        paths.extend(p for p in d.rglob("*") if p.is_file() and p.suffix in (".py",".json",".md"))
    # First-round trained weights are hash-checked as bytes, never deserialized or reused.
    first=ROOT/"animalact2/first_round"
    inventory=json.loads((first/"FINAL_VERIFICATION.json").read_text(encoding="utf-8"))["artifacts"]
    paths.extend(first/name for name in inventory)
    paths.extend([SOURCE,*(Path(json.loads(SOURCE.read_text())["path"])/n for n in ("config.json","model.safetensors","preprocessor_config.json"))])
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths)) if "__pycache__" not in p.parts}


def capture():
    p=OUT/"protection_baseline.json"
    if p.exists():return
    write_json(p,dict(started_unix=time.time(),metadata=metadata_snapshot(),SHA256=protected_hashes()))


def verify():
    baseline=json.loads((OUT/"protection_baseline.json").read_text(encoding="utf-8"))
    assert metadata_snapshot()==baseline["metadata"],"Historical metadata changed"
    assert protected_hashes()==baseline["SHA256"],"Protected content changed"
    result=dict(status="passed",metadata_files=len(baseline["metadata"]),content_hashed_files=len(baseline["SHA256"]),
                release_originals_ResNet_first_round_and_pretrained_weights_unchanged=True,Test_content_read=False)
    write_json(OUT/"protection_verification.json",result)
    return result


if __name__=="__main__":
    # Capture needs read-only byte hashing of historical checkpoints; the runtime guard
    # intentionally rejects deserialization paths and is enabled by the actual pilot.
    if sys.argv[1:]==["verify"]:print(json.dumps(verify()))
    else:capture();print("phase2_2 protection baseline captured")
