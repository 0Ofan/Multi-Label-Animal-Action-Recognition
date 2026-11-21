"""Validate the immutable full-cache inventory and fixed experiment protocol."""
import json
import time
from .support import ROOT,OUT,FULL,enable_guard,write_json,sha


def check():
    from animalact2.phase1_5.config import load_config,cohort_hash,config_hash
    start=time.perf_counter()
    integrity=json.loads((FULL/"artifact_manifest.json").read_text(encoding="utf-8"))
    checked=0;cache_files=0
    for rel,row in integrity["files"].items():
        p=(FULL/rel).resolve()
        assert p.is_relative_to(FULL) and p.stat().st_size==row["bytes"] and sha(p)==row["sha256"],rel
        checked+=1;cache_files+=rel.startswith("cache/")
    status=json.loads((FULL/"validation_summary.json").read_text(encoding="utf-8"))
    assert status["status"]=="passed" and status["verified_clips"]==5646 and status["verified_unique_frames"]==73884
    configs={name:load_config(ROOT/f"animalact2/phase1_5/configs/{name}_seed42_bce.json") for name in ("model1","model2","model3")}
    for name,c in configs.items():
        assert c["model"]==name and c["seed"]==42 and c["loss"]["name"]=="bce"
        assert c["data"]["batch_size"]==8 and c["data"]["valid_batch_size"]==8 and c["data"]["accumulation_steps"]==4
        assert c["optimizer"]==dict(name="AdamW",lr=.001,weight_decay=.0001,grad_clip_norm=5.)
        assert c["scheduler"]==dict(name="warmup_cosine",warmup_epochs=2,min_lr_ratio=.1)
        assert c["training"]["max_epochs"]==50 and c["training"]["patience"]==10 and c["backbone"]["frozen"] is True
    assert len({cohort_hash(c) for c in configs.values()})==1
    result=dict(status="passed",files_SHA_checked=checked,cache_files_checked=cache_files,Train_clips=4667,Valid_clips=979,unique_frames=73884,
                config_SHA256={n:config_hash(c) for n,c in configs.items()},shared_cohort_SHA256=cohort_hash(configs["model1"]),
                cache_artifact_manifest_SHA256=sha(FULL/"artifact_manifest.json"),cache_context_SHA256=sha(FULL/"cache/context.json"),
                seconds=time.perf_counter()-start,Test_read=False,CNN_frames_extracted=0)
    write_json(OUT/"preflight/cache_and_protocol.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=="__main__":
    # Import Torch before the filesystem audit hook (its Windows platform probe opens NUL).
    import torch
    enable_guard();check()
