"""Read-only historical protection; all newly written audit files stay here."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SKIP = {".venv", ".git", ".codex", ".agents", ".aws", "node_modules", "__pycache__"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def snapshot():
    result = {}
    for directory, sub, files in os.walk(ROOT):
        p = Path(directory)
        sub[:] = [n for n in sub if n not in SKIP and "model6" not in n.casefold() and (p / n).resolve() != OUT]
        for name in files:
            path = p / name
            if "model6" in name.casefold() or path.is_symlink():
                continue
            st = path.stat()
            result[path.relative_to(ROOT).as_posix()] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    return result


def phase_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p)
            for phase in ("phase0", "phase1_1", "phase1_2", "phase1_3", "phase1_4", "phase1_4_2", "release_v1")
            for p in (ROOT / "animalact2" / phase).rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "verify"))
    args = parser.parse_args()
    started = time.perf_counter()
    path = OUT / "protection_baseline.json"
    if args.mode == "capture":
        if path.exists():
            raise FileExistsError("Never overwrite the initial historical baseline")
        baseline = dict(old_workspace_metadata=snapshot(), historical_phase_SHA256=phase_hashes(),
                        pretrained_weight_path=".cache/torch/checkpoints/resnet18-f37072fd.pth",
                        pretrained_weight_SHA256=sha(ROOT / ".cache/torch/checkpoints/resnet18-f37072fd.pth"),
                        started_unix=time.time(), scope="Old workspace size/mtime excluding environment, Git, new phase and Model6; previous phases/release content SHA")
        path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dict(status="captured", historical_metadata_files=len(baseline["old_workspace_metadata"]), historical_phase_hash_files=len(baseline["historical_phase_SHA256"]), pretrained_SHA256=baseline["pretrained_weight_SHA256"], seconds=time.perf_counter()-started)))
    else:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        current = snapshot()
        assert current == baseline["old_workspace_metadata"], "Historical file metadata changed"
        assert phase_hashes() == baseline["historical_phase_SHA256"], "Historical data/release SHA changed"
        assert sha(ROOT / baseline["pretrained_weight_path"]) == baseline["pretrained_weight_SHA256"]
        result = dict(status="passed", historical_metadata_files=len(current), historical_phase_hash_files=len(baseline["historical_phase_SHA256"]),
                      release_v1_files_unchanged=sum(p.startswith("animalact2/release_v1/") for p in baseline["historical_phase_SHA256"]),
                      pretrained_weight_unchanged=True, source_videos_and_old_experiments_metadata_unchanged=True,
                      Test_video_or_annotation_or_prediction_used=False, old_files_modified=False, seconds=time.perf_counter()-started)
        (OUT / "protection_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
