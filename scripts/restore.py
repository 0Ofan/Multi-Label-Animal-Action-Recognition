"""Download verified runtime or historical 2.0 checkpoints. Never the raw dataset."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from urllib.request import urlopen,Request
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("kind",choices=["runtime","final","all","checkpoints","research"]);args=ap.parse_args()
    if args.kind in ("final", "all"):
        import sys;sys.path.insert(0,str(ROOT))
        from deployment.final_artifacts import ensure_final_runtime
        ensure_final_runtime()
        if args.kind == "final":
            print("Final integrated model files verified");return
    if args.kind=="research":
        import sys,subprocess
        subprocess.run([sys.executable,str(ROOT/"research_tools/restore.py")],check=True)
        return
    if args.kind in ("runtime", "all"):
        import sys;sys.path.insert(0,str(ROOT))
        from deployment.artifacts import ensure_runtime
        ensure_runtime();print("Runtime model files verified");return
    info=json.loads((ROOT/"deployment/checkpoint_archive.json").read_text());tmp=ROOT/".downloads";tmp.mkdir(exist_ok=True)
    archive=tmp/"checkpoints.zip"
    with urlopen(Request(info["url"],headers={"User-Agent":"AnimalAct2-restore"}),timeout=300) as src,archive.open("wb") as dst:
        while chunk:=src.read(1024*1024):dst.write(chunk)
    assert archive.stat().st_size==info["bytes"] and sha(archive)==info["SHA256"]
    expected={row["path"]:row for row in info["files"]}
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(expected)
        for name in z.namelist():
            dest=(ROOT/name).resolve()
            if not dest.is_relative_to(ROOT/"animalact2"):raise ValueError("Unsafe archive entry")
            digest=expected[name]["SHA256"]
            if dest.exists():
                if sha(dest)!=digest:raise FileExistsError("Different existing checkpoint: "+name)
                continue
            dest.parent.mkdir(exist_ok=True,parents=True)
            pending=dest.with_suffix(".pending")
            with z.open(name) as src,pending.open("wb") as dst:
                while chunk:=src.read(1024*1024):dst.write(chunk)
            if sha(pending)!=digest:raise ValueError("Checkpoint checksum mismatch")
            pending.replace(dest)
    print(f"Verified {len(expected)} original AnimalAct 2.0 checkpoints")
if __name__=="__main__":main()
