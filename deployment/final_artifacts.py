"""Restore the final team's original checkpoints from a verified release.

Metadata and evaluation pages work without downloading or loading a model.
Duplicate checkpoint bytes are stored once in the archive. Existing differing
checkpoints are rejected rather than replaced.
"""
import hashlib
import json
import os
import tempfile
import threading
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "deployment/final_model_manifest.json"
_LOCK = threading.RLock()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def declared(path):
    relative = Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    return relative in manifest()["files"]


def ensure_final_runtime():
    with _LOCK:
        data = manifest()
        pending = []
        for name, info in data["files"].items():
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT.resolve()) or Path(name).parts[0] not in ("proposal", "ensemble_integration"):
                raise ValueError("Unsafe checkpoint destination")
            if target.exists():
                if not target.is_file() or target.stat().st_size != info["bytes"] or sha(target) != info["SHA256"]:
                    raise ValueError(f"Existing checkpoint differs from final release; not overwritten: {name}")
            else:
                pending.append(name)
        if not pending:
            return
        folder = ROOT / ".runtime"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="final-models-", dir=folder) as temp:
            archive = Path(temp) / "models.zip"
            source = data["archive"]
            with urlopen(Request(source["url"], headers={"User-Agent": "AnimalAct-final-runtime"}), timeout=300) as response, archive.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if archive.stat().st_size != source["bytes"] or sha(archive) != source["SHA256"]:
                raise ValueError("Final model archive checksum mismatch")
            expected = {row["blob"] for row in data["files"].values()}
            with zipfile.ZipFile(archive) as bundle:
                if set(bundle.namelist()) != expected or len(bundle.namelist()) != len(expected):
                    raise ValueError("Unexpected final model archive entries")
                # Validate the complete download before committing any checkpoint.
                for blob in expected:
                    raw = bundle.read(blob)
                    rows = [row for row in data["files"].values() if row["blob"] == blob]
                    if any(len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["SHA256"] for row in rows):
                        raise ValueError("Final model payload checksum mismatch")
                for name in pending:
                    target = ROOT / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(prefix=".restore-", dir=target.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(bundle.read(data["files"][name]["blob"]))
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        if target.exists():
                            if sha(target) != data["files"][name]["SHA256"]:
                                raise ValueError("Checkpoint changed during restoration")
                        else:
                            temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_final_runtime()
    print("Final team model checkpoints verified")
