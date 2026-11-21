"""Phase-local writes and read-only historical protection; Test sources are blocked."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
READS = set()


def forbidden(path):
    low = "/" + path.as_posix().casefold()
    return any(s in low for s in ("/videos/test/", "/annotations/test/", "/test_metadata_only/", "predictions", "model6"))


def guard(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("No network operations in Phase1.5")
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
    if write and not path.is_relative_to(OUT):
        raise RuntimeError("All new files must stay in phase1_5: " + str(path))
    if path.is_relative_to(ROOT):
        if forbidden(path) and not path.is_relative_to(OUT):
            raise RuntimeError("Test/prediction/Model6 source read blocked: " + str(path))
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_relative_to(OUT) and path.suffix.lower() in (".pt", ".pth", ".npy", ".npz", ".safetensors") and rel != ".cache/torch/checkpoints/resnet18-f37072fd.pth":
            raise RuntimeError("Old task weights/feature cache read blocked")
        if not write:
            READS.add(rel)


def enable_guard():
    sys.addaudithook(guard)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT):
        raise ValueError("Output outside phase1_5")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def snapshot():
    result = {}
    skip = {".venv", ".git", ".codex", ".agents", ".aws", "node_modules", "__pycache__"}
    for directory, sub, files in os.walk(ROOT):
        p = Path(directory)
        sub[:] = [n for n in sub if n not in skip and "model6" not in n.casefold() and (p / n).resolve() != OUT]
        for name in files:
            path = p / name
            if "model6" in name.casefold() or path.is_symlink():
                continue
            st = path.stat()
            result[path.relative_to(ROOT).as_posix()] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    return result


def content_hashes():
    # Do not open Test manifests/labels even for checksums. Their metadata is protected above.
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for d in (ROOT / "animalact2").iterdir() if d.is_dir() and d.resolve() != OUT
            for p in d.rglob("*") if p.is_file() and not forbidden(p)}


def capture():
    path = OUT / "protection_baseline.json"
    if path.exists():
        raise FileExistsError("Initial protection baseline already exists")
    obj = dict(captured_unix=time.time(), metadata=snapshot(), content_SHA256=content_hashes(),
               official_weights_SHA256=sha(ROOT / ".cache/torch/checkpoints/resnet18-f37072fd.pth"),
               Test_content_opened=False, Test_protection="file size/mtime only; Test content never opened")
    write_json(path, obj)
    print(json.dumps(dict(status="captured", metadata_files=len(obj["metadata"]), hashed_non_Test_historical_files=len(obj["content_SHA256"]))))


def verify():
    obj = json.loads((OUT / "protection_baseline.json").read_text(encoding="utf-8"))
    assert snapshot() == obj["metadata"], "Historical metadata changed"
    assert content_hashes() == obj["content_SHA256"], "Historical non-Test content changed"
    assert sha(ROOT / ".cache/torch/checkpoints/resnet18-f37072fd.pth") == obj["official_weights_SHA256"]
    result = dict(status="passed", metadata_files_unchanged=len(obj["metadata"]), hashed_non_Test_files_unchanged=len(obj["content_SHA256"]),
                  phase1_5_1_and_release_unchanged=True, Test_content_opened=False, official_pretrained_weights_unchanged=True)
    write_json(OUT / "protection_verification.json", result)
    return result


if __name__ == "__main__":
    enable_guard()
    if sys.argv[1:] == ["capture"]:
        capture()
    elif sys.argv[1:] == ["verify"]:
        print(json.dumps(verify(), indent=2))
    else:
        raise SystemExit("Use capture or verify")
