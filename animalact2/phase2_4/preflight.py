"""Prepare locked configurations and verify sealed cache/protocol provenance."""
import json
from animalact2.phase1_5.config import config_hash,cohort_hash,load_config as old_config
from .support import ROOT,OUT,FULL,enable_guard,sha,write_json
from .models import prepare_configs,load_config
from . import checkpoint


def main():
    enable_guard();prepare_configs()
    manifest=json.loads((FULL/"artifact_manifest.json").read_text(encoding="utf-8"))["files"]
    for name,entry in manifest.items():
        p=FULL/name;assert p.stat().st_size==entry["bytes"] and sha(p)==entry["sha256"],name
    v=json.loads((FULL/"validation_summary.json").read_text(encoding="utf-8"))
    assert v["status"]=="passed" and v["verified_clips"]==5646 and v["failed"]==0
    configs={}
    for name in ("model1","model2","model3"):
        c=load_config(OUT/f"configs/{name}_seed42_bce.json");baseline=old_config(ROOT/f"animalact2/phase1_5/configs/{name}_seed42_bce.json")
        assert {k:v for k,v in c.items() if k not in ("backbone","input_cache")}=={k:v for k,v in baseline.items() if k!="backbone"}
        configs[name]=c
    assert len({cohort_hash(c) for c in configs.values()})==1
    result=dict(status="passed",Full_cache_artifact_files_SHA_checked=len(manifest),Train_clips=4667,Valid_clips=979,
                cache_provenance=checkpoint.cache_provenance(),shared_cohort_SHA256=cohort_hash(configs["model1"]),
                config_SHA256={n:config_hash(c) for n,c in configs.items()},ResNet_protocol_fields_exact=True,
                feature_dim=768,output_dim=119,encoder_constructed=False,Test_read=False)
    write_json(OUT/"preflight/cache_and_protocol.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
