"""Final read-only evidence checks, environment capture and local artifact inventory."""
from __future__ import annotations
import ast
import io
import json
from pathlib import Path
import sys
import time
import unittest
import cv2
import numpy as np
from PIL import __version__ as pillow_version
import torch
import torchvision
from .safety import ROOT,OUT,enable_guard,verify,sha,write_json
from .config import load_config,config_hash,cohort_hash,development_fingerprints
from .checkpoint import implementation_fingerprints
from .tests import test_final_plan


def main():
    started=time.perf_counter();enable_guard()
    if (OUT/"FINAL_VERIFICATION.json").exists():raise FileExistsError("Preserve completed preparation evidence")
    audit=json.loads((OUT/"data_audit/summary.json").read_text(encoding="utf-8"))
    validation=json.loads((OUT/"preparation_validation.json").read_text(encoding="utf-8"))
    plan=json.loads((OUT/"first_round_plan.json").read_text(encoding="utf-8"))
    assert audit["status"] == validation["status"] == "passed" and audit["clips_checked"]==5646 and audit["failures"]==0
    assert validation["unit_tests"]==29 and len(validation["real_BCE_ASL_cases"])==6 and validation["same_input_across_three_models_and_losses"]
    assert not audit["Test_opened"] and not validation["Test_opened"] and not validation["formal_training"]
    assert not (OUT/"runs").exists(),"Formal experiment directory must not have been created"
    stream=io.StringIO()
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_final_plan))
    (OUT/"final_plan_test_results.txt").write_text(stream.getvalue(),encoding="utf-8")
    assert tests.wasSuccessful() and tests.testsRun==2
    configs=[load_config(OUT/f"configs/{name}_seed42_bce.json") for name in ("model1","model2","model3")]
    assert len({cohort_hash(c) for c in configs})==1
    assert all(c["training"]["max_epochs"]==50 and c["training"]["patience"]==10 for c in configs)
    matrix=json.loads((OUT/"configs/experiment_matrix.json").read_text(encoding="utf-8"))
    assert matrix["shared_cohort_SHA256"]==cohort_hash(configs[0]) and not matrix["automatic_execution"]
    assert plan["max_epochs"]==50 and plan["patience"]==10 and plan["optimizer_updates_per_epoch"]==146
    data_sha,code_sha=development_fingerprints(),implementation_fingerprints()
    checked_checkpoints=[]
    for path in sorted((OUT/"diagnostics").rglob("*.pt")):
        state=torch.load(path,map_location="cpu",weights_only=True)
        assert state["diagnostic"] is True and state["data_fingerprints"]==data_sha and state["implementation_fingerprints"]==code_sha
        assert state["backbone"]["sha256"]=="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
        assert state["class_policy"]["unsupported_ids"].tolist()==[23]
        assert state["config_SHA256"]==config_hash(state["config"])
        checked_checkpoints.append(path.relative_to(OUT).as_posix())
    assert len(checked_checkpoints)==7
    parsed=[]
    for path in sorted(OUT.rglob("*.py")):
        tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.level==0:
                assert (node.module or "").split(".")[0] not in {"models","features","training","inference","transformers","frontend"}
        parsed.append(path.relative_to(OUT).as_posix())
    environment=dict(python=sys.version,executable=sys.executable,platform=sys.platform,Windows_version=list(sys.getwindowsversion()),torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                     cuda_runtime=torch.version.cuda,opencv=cv2.__version__,pillow=pillow_version,numpy=np.__version__,GPU=torch.cuda.get_device_name(0),GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                     precision="FP32",dependencies_installed=False,global_environment_modified=False,source_weights_downloaded=False)
    write_json(OUT/"environment_snapshot.json",environment)
    protection=verify()
    inventory={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("artifact_manifest.json","FINAL_VERIFICATION.json")}
    write_json(OUT/"artifact_manifest.json",dict(files=inventory,excluded_self_references=["artifact_manifest.json","FINAL_VERIFICATION.json"]))
    final=dict(status="ready_for_human_review_no_formal_training",phase="1.5",release_id="animalact2-2.0.0-release-v1",
               full_Train_clips_verified=4667,full_Valid_clips_verified=979,parent_videos_verified=2505,data_failures=0,dense_source_frames_decoded=326795,unique_sampled_frames_RGB_resize_verified=73884,
               total_distinct_automated_tests_passed=31,prior_contract_and_framework_tests=29,final_plan_tests=2,real_model_loss_cases=6,real_data_diagnostic_updates=9,
               diagnostic_checkpoint_files_checked=checked_checkpoints,diagnostic_checkpoints_data_and_code_fingerprints_current=True,
               first_round=dict(models=[c["model"] for c in configs],seed=42,loss="bce",max_epochs=50,patience=10,physical_batch=8,accumulation_steps=4,updates_per_epoch=146,shared_cohort_SHA256=cohort_hash(configs[0])),
               maximum_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in validation["pipeline_benchmarks"]),maximum_peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in validation["pipeline_benchmarks"]),
               estimated_all_models_max_budget_hours=plan["nominal_three_models_max_budget_wall_hours"],data_audit_seconds=audit["elapsed_seconds"],bounded_diagnostic_seconds=validation["total_seconds"],
               Python_files_AST_checked=parsed,historical_protection=protection,finalization_seconds=time.perf_counter()-started,
               Test_video_annotation_prediction_or_manifest_read=False,formal_training_started=False,full_RGB_or_feature_cache_created=False,
               original_or_historical_files_modified=False,Dashboard_developed=False,model_architecture_changed=False,extra_research_model_added=False,
               stopped_for_review=True,automatic_training_or_followup=False,new_files=0,total_bytes=0,MiB=0)
    for _ in range(10):
        write_json(OUT/"FINAL_VERIFICATION.json",final)
        files=[p for p in OUT.rglob("*") if p.is_file()];actual=len(files),sum(p.stat().st_size for p in files)
        if actual==(final["new_files"],final["total_bytes"]):break
        final["new_files"],final["total_bytes"]=actual;final["MiB"]=round(actual[1]/1024**2,6)
    else:raise RuntimeError("Could not stabilize artifact sizes")
    assert all(sha(OUT/path)==row["sha256"] for path,row in inventory.items())
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":main()
