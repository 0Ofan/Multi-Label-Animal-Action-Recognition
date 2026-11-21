"""Restore the complete sealed 2.0 research snapshot into an isolated directory."""
from pathlib import Path
import argparse,json,hashlib,zipfile,shutil,uuid
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def restore(target,archive=None):
    target=Path(target).resolve()
    if target==ROOT or target.is_relative_to(ROOT/"animalact2"):raise ValueError("Use an isolated research workspace, never the website source")
    info=json.loads((ROOT/"deployment/research_archive.json").read_text(encoding="utf-8"))
    if archive is None:
        downloads=ROOT/".downloads";downloads.mkdir(exist_ok=True);archive=downloads/"research-snapshot.zip"
        if not archive.exists() or sha(archive)!=info["SHA256"]:
            pending=archive.with_suffix(".pending")
            with urlopen(Request(info["url"],headers={"User-Agent":"AnimalAct2-research-restore"}),timeout=300) as src,pending.open("wb") as dst:
                while chunk:=src.read(1024*1024):dst.write(chunk)
            if pending.stat().st_size!=info["bytes"] or sha(pending)!=info["SHA256"]:raise ValueError("Research archive checksum mismatch")
            pending.replace(archive)
    archive=Path(archive)
    if archive.stat().st_size!=info["bytes"] or sha(archive)!=info["SHA256"]:raise ValueError("Wrong archive")
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read("RESEARCH_FILES.json"));expected=manifest["files"]
        if len(z.namelist())!=len(expected)+1 or set(z.namelist())!=set(expected)|{"RESEARCH_FILES.json"}:raise ValueError("Unexpected ZIP entries")
        for i,(name,row) in enumerate(expected.items()):
            dest=(target/name).resolve()
            if not dest.is_relative_to(target):raise ValueError("Unsafe ZIP path")
            if dest.exists():
                if dest.stat().st_size!=row["bytes"] or sha(dest)!=row["SHA256"]:raise FileExistsError("Different existing research file: "+name)
                continue
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+"."+uuid.uuid4().hex+".pending")
            with z.open(name) as src,tmp.open("wb") as dst:
                while chunk:=src.read(1024*1024):dst.write(chunk)
            if tmp.stat().st_size!=row["bytes"] or sha(tmp)!=row["SHA256"]:raise ValueError("Extracted checksum mismatch: "+name)
            tmp.replace(dest)
            if (i+1)%5000==0:print(f"Verified restored files {i+1}/{len(expected)}",flush=True)
        (target/"RESEARCH_FILES.json").write_text(json.dumps(manifest,ensure_ascii=False),encoding="utf-8")
    # Supplement the selected 2.0 data-foundation sources; old-model Phase0 audits are intentionally absent.
    foundation=json.loads((ROOT/"deployment/data_foundation_files.json").read_text(encoding="utf-8"))
    for name,row in foundation.items():
        src=ROOT/name;dest=target/name
        if sha(src)!=row["SHA256"]:raise ValueError("Foundation source checksum mismatch")
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha(dest)!=row["SHA256"]:raise FileExistsError(dest)
        if not dest.exists():shutil.copyfile(src,dest)
    relative=next(n.rsplit("/",1)[0] for n in expected if n.endswith("/model.safetensors"))
    (target/"research_locations.json").write_text(json.dumps(dict(videomae_directory=relative,raw_data_not_included=True),indent=2),encoding="utf-8")
    print(json.dumps(dict(status="restored_verified",files=len(expected),target=str(target),raw_data_not_included=True)),flush=True)
    return target
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--target",type=Path,default=ROOT/"research_workspace");p.add_argument("--archive",type=Path)
    a=p.parse_args();restore(a.target,a.archive)
