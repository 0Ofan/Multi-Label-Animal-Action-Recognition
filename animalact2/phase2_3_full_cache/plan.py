"""Development-only plan and Pilot provenance gate; no Test source access."""
import csv
import json
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from .support import ROOT,OUT,PILOT,RELEASE,sha,digest,write_json
from .storage import cache_context


def build_plan():
    ctx=cache_context()
    assert ctx==json.loads((PILOT/"cache/context.json").read_text(encoding="utf-8")),"Pilot context/library/implementation changed"
    for name in ("pilot_validation.json","independent_validation.json","adapter_validation.json"):
        assert json.loads((PILOT/name).read_text(encoding="utf-8"))["status"]=="passed"
    datasets={s:ReleaseClipDataset(s) for s in ("train","valid")}
    assert tuple(len(datasets[s]) for s in datasets)==(4667,979)
    expected=dict(context=ctx,manifest_SHA256={s:ds.manifest_sha256 for s,ds in datasets.items()},
                  selections={s:[r["clip_id"] for r in ds.records] for s,ds in datasets.items()},
                  expected_real_frame_references=81641,expected_time_steps=41156,Pilot_clips=120,Test_enabled=False)
    path=OUT/"full_plan.json"
    if path.exists():assert json.loads(path.read_text(encoding="utf-8"))==expected,"Full plan changed"
    else:write_json(path,expected)
    return expected,datasets


def verify_sources(datasets):
    checks={}
    for split,ds in datasets.items():
        for r in ds.records:
            for kind in ("video","annotation"):
                rel=r[kind+"_path"]
                if rel in checks:
                    assert checks[rel]["sha256"]==r[kind+"_sha256"]
                    continue
                p=ROOT/rel;actual=sha(p)
                if actual!=r[kind+"_sha256"]:raise ValueError("Source content differs from release: "+rel)
                st=p.stat();checks[rel]=dict(split=split,kind=kind,path=rel,sha256=actual,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    p=OUT/"source_identity.json"
    if p.exists():assert json.loads(p.read_text(encoding="utf-8"))==checks,"Source identity changed since initial build"
    else:write_json(p,checks)
    with (OUT/"source_checks.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(next(iter(checks.values()))));w.writeheader();w.writerows(checks.values())
    return checks
