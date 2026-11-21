"""Verified public model assets, streamed to disk; no raw training dataset."""
from pathlib import Path
import hashlib,json,os,tempfile,threading,zipfile
from urllib.request import urlopen,Request
ROOT=Path(__file__).resolve().parents[1]
_LOCK=threading.RLock()
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def manifest():return json.loads((ROOT/"deployment/model_manifest.json").read_text(encoding="utf-8"))
def ensure_runtime():
    with _LOCK:
        m=manifest(); expected=dict(m["runtime_files"])
        if all((ROOT/name).is_file() and sha(ROOT/name)==digest for name,digest in expected.items()):return
        folder=ROOT/".runtime";folder.mkdir(exist_ok=True)
        archive=folder/"runtime-download.zip"
        source=m["runtime_archive"]
        with urlopen(Request(source["url"],headers={"User-Agent":"AnimalAct2-runtime"}),timeout=300) as response, archive.open("wb") as f:
            while chunk:=response.read(1024*1024):f.write(chunk)
        if archive.stat().st_size!=source["bytes"] or sha(archive)!=source["SHA256"]:raise ValueError("Model archive checksum mismatch")
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist())!=set(expected):raise ValueError("Unexpected archive entries")
            for name in z.namelist():
                target=(ROOT/name).resolve()
                if not target.is_relative_to((ROOT/"runtime_models").resolve()):raise ValueError("Unsafe model archive path")
                target.parent.mkdir(parents=True,exist_ok=True)
                pending=target.with_suffix(target.suffix+".pending")
                with z.open(name) as src,pending.open("wb") as dst:
                    while chunk:=src.read(1024*1024):dst.write(chunk)
                if sha(pending)!=expected[name]:raise ValueError("Extracted model checksum mismatch")
                pending.replace(target)
        archive.unlink()  # Only this downloader's temporary archive, never dataset files.
def resolved_source():
    ensure_runtime();source=dict(manifest()["source"]);source["path"]=str(ROOT/"runtime_models/videomae");return source
