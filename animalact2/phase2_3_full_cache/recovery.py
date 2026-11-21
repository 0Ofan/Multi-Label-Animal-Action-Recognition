"""Preserve incomplete-run evidence and validate committed arrays before resumption."""
import json
from .support import OUT,PILOT,enable_guard,write_json
from .plan import build_plan
from .storage import CACHE,read_cache


def main():
    enable_guard()
    target=OUT/"interruption_recovery.json"
    if target.exists():
        print("Original interruption evidence already preserved",flush=True);return
    if (OUT/"extraction_summary.json").exists():raise RuntimeError("No unfinished full run to reconcile")
    original=json.loads((OUT/"progress.json").read_text(encoding="utf-8"))
    assert original["stage"]=="extracting"
    plan,datasets=build_plan()
    pilot=json.loads((PILOT/"pilot_plan.json").read_text(encoding="utf-8"))
    pilot_ids={r["clip_id"] for rows in pilot["records"].values() for r in rows}
    counts={};copied=0;keys=set()
    for split,ds in datasets.items():
        count=0
        for r in ds.records:
            item=read_cache(r,plan["context"])
            if item is not None:
                count+=1;copied+=r["clip_id"] in pilot_ids
                assert item[1]["key"] not in keys;keys.add(item[1]["key"])
        counts[split]=count
    assert not list(CACHE.rglob(".pending_*"))
    assert original["reused_full"]==0 and copied==original["Pilot_copied"]==120
    assert all(counts[s]>=original["completed"][s] for s in counts)
    result=dict(status="committed_cache_verified",original_last_progress=original,
                verified_committed_counts=counts,verified_committed_total=sum(counts.values()),
                original_Pilot_copied=copied,original_new_committed=sum(counts.values())-copied,
                remaining_clips=5646-sum(counts.values()),recorded_elapsed_lower_bound_seconds=original["seconds"],
                original_final_GPU_peak_available=False,original_final_RAM_peak_available=False,
                original_final_run_summary_available=False,exit_cause="not established from persisted records",
                scope="Elapsed time is a persisted lower bound, not an exact original-run total; no feature recomputation",Test_read=False)
    write_json(target,result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
