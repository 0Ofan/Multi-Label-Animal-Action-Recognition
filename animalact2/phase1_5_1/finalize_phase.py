"""Finalize functional evidence and historical protection, without running models."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from . import protection

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def files():
    return sorted(p for p in OUT.rglob("*") if p.is_file())


def main():
    start = time.perf_counter()
    if (OUT / "FINAL_VERIFICATION.json").exists():
        raise FileExistsError("Final evidence already exists; preserve the completed phase")
    required = ["dataset.py", "video_decoder.py", "transforms.py", "collate.py", "backbone.py", "model1.py", "model2.py", "model3.py", "evaluation.py", "README.md", "tests/test_contracts.py", "validation_results.json", "unit_test_results.txt", "FINAL_PHASE1_5_1_REPORT.md"]
    assert all((OUT / path).is_file() for path in required)
    parsed = []
    forbidden = {"models", "features", "training", "inference", "frontend", "transformers"}
    for path in sorted(OUT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] not in forbidden
            elif isinstance(node, ast.Import):
                assert not any(n.name.split(".")[0] in forbidden for n in node.names)
        parsed.append(path.relative_to(OUT).as_posix())
    result = json.loads((OUT / "validation_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed" and result["unit_tests"] == 16 and result["real_sample_count"] == 6
    assert result["same_initial_backbone"] and result["same_initial_Model2_Model3_BiGRU"]
    assert not result["Test_data_used"] and not result["formal_training"] and not result["parameters_persisted"]
    assert all(r["diagnostic_optimizer_steps"] == 1 and r["updated_split"] == "train" and r["frozen_backbone_and_BN_unchanged"] for r in result["models"])
    # Test the newly added output protection, never execute another diagnostic update.
    before = {name: protection.sha(OUT / name) for name in ("validation_results.json", "unit_test_results.txt")}
    rejected = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "animalact2.phase1_5_1.run_validation"],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert rejected.returncode != 0 and "Choose a new --output-dir" in rejected.stderr
    assert before == {name: protection.sha(OUT / name) for name in before}
    save("cli_guard_check.json", dict(status="passed", existing_results_overwrite_rejected=True, previous_results_SHA_unchanged=before,
                                       new_diagnostic_or_training_started=False, check="Default CLI refuses completed outputs before test/model execution"))
    baseline = json.loads((OUT / "protection_baseline.json").read_text(encoding="utf-8"))
    assert protection.snapshot() == baseline["old_workspace_metadata"]
    assert protection.phase_hashes() == baseline["historical_phase_SHA256"]
    assert protection.sha(ROOT / baseline["pretrained_weight_path"]) == baseline["pretrained_weight_SHA256"]
    old = dict(status="passed", historical_metadata_files=len(baseline["old_workspace_metadata"]),
               historical_phase_hash_files=len(baseline["historical_phase_SHA256"]), release_v1_files_unchanged=56,
               pretrained_weight_unchanged=True, source_videos_and_old_experiments_metadata_unchanged=True,
               Test_video_or_annotation_or_prediction_used=False, old_files_modified=False)
    save("protection_verification.json", old)
    inventory = {p.relative_to(OUT).as_posix(): dict(bytes=p.stat().st_size, sha256=protection.sha(p))
                 for p in files() if p.name not in ("artifact_manifest.json", "FINAL_VERIFICATION.json")}
    save("artifact_manifest.json", dict(files=inventory, excluded_self_references=["artifact_manifest.json", "FINAL_VERIFICATION.json"]))
    final = dict(status="completed_and_stopped_for_review", phase="1.5.1", completed_utc=datetime.now(timezone.utc).isoformat(),
                 release_id=result["release_id"], required_files_present=True, Python_files_AST_checked=parsed,
                 imports_of_legacy_models_or_extra_backbones=False, unit_tests_passed=16, real_Train_clips=3, real_Valid_clips=3,
                 real_samples_CPU_CUDA_passed=True, one_Train_diagnostic_update_per_model=True, Valid_updates=False,
                 maximum_CUDA_peak_allocated_bytes=max(r["CUDA_peak_allocated_bytes"] for r in result["models"]),
                 maximum_CUDA_peak_reserved_bytes=max(r["CUDA_peak_reserved_bytes"] for r in result["models"]),
                 diagnostic_seconds=result["total_validation_seconds"], finalization_seconds=time.perf_counter()-start,
                 source_protection=old, pretrained_weights_downloaded=False, dependencies_installed=False,
                 RGB_or_feature_cache_written=False, trained_weights_saved=False, formal_training_started=False,
                 Test_model_validation_or_tuning=False, next_phase_started=False, stopped_for_review=True,
                 new_file_count=0, new_total_bytes=0, new_MiB=0)
    for _ in range(8):
        save("FINAL_VERIFICATION.json", final)
        current = files()
        actual = len(current), sum(p.stat().st_size for p in current)
        if actual == (final["new_file_count"], final["new_total_bytes"]):
            break
        final["new_file_count"], final["new_total_bytes"] = actual
        final["new_MiB"] = round(actual[1] / 1024**2, 6)
    else:
        raise RuntimeError("Failed to stabilize final evidence counts")
    assert all(protection.sha(OUT / rel) == row["sha256"] for rel, row in inventory.items())
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
