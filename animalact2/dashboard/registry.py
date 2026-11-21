"""Portable model registry exported from verified Phase2.4/2.5, no raw data."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RELEASE=ROOT/"animalact2/release_v1"
def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def model_info(key="attention"):
    manifest=read_json(ROOT/"deployment/model_manifest.json")
    if key not in manifest["models"]:raise ValueError("Unknown model")
    info=dict(manifest["models"][key]);info["checkpoint"]=str(ROOT/info["checkpoint_path"])
    return info
def classes():
    rows=read_json(ROOT/"deployment/model_manifest.json")["classes"]
    if [r["class_id"] for r in rows]!=list(range(119)):raise ValueError("Invalid vocabulary")
    return rows
