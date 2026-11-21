"""Build the fixed Train/Valid cache with the unchanged validated Pilot encoder."""
from __future__ import annotations
import gc
import json
import sys
import time
import numpy as np
import torch
from animalact2.phase2_2.adapter import FrozenVideoMAEAdapter,encoder_state_hash
from .support import OUT,PILOT,enable_guard,write_json,RAMMonitor
from .storage import CACHE,read_cache,commit,reuse_pilot
from .plan import build_plan,verify_sources


def main():
    enable_guard();torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest");torch.backends.cudnn.allow_tf32=False
    resume_check=sys.argv[1:]==["--resume-check"]
    if sys.argv[1:] and not resume_check:raise ValueError("Only the optional --resume-check flag is supported")
    if not (OUT/"protection_baseline.json").exists():raise RuntimeError("Capture original/Pilot integrity baseline first")
    run_path=OUT/"runs"/f"run_{len(list((OUT/'runs').glob('run_*.json')))+1:04d}.json"
    started=time.perf_counter();counts={s:0 for s in ("train","valid")};new=copied=reused=decoded=0;failed=[]
    with RAMMonitor() as ram:
        plan,datasets=build_plan();ctx=plan["context"]
        if (CACHE/"context.json").exists():assert json.loads((CACHE/"context.json").read_text())==ctx
        else:write_json(CACHE/"context.json",ctx)
        sources=verify_sources(datasets)
        print(json.dumps(dict(stage="source_identity_passed",source_files=len(sources),Train=4667,Valid=979)),flush=True)
        pending={s:[] for s in datasets}
        for split,ds in datasets.items():
            for i,r in enumerate(ds.records):
                if read_cache(r,ctx) is not None:reused+=1;counts[split]+=1
                elif resume_check:raise RuntimeError("Resume verification requires a complete cache: "+r["clip_id"])
                elif reuse_pilot(r,ctx):copied+=1;counts[split]+=1
                else:pending[split].append(i)
        print(json.dumps(dict(stage="cache_discovery",reused_full=reused,copied_Pilot=copied,pending=sum(map(len,pending.values())))),flush=True)
        adapter=None;before=None;batch_times=[];allocated=reserved=0;extraction_start=time.perf_counter()
        if any(pending.values()):
            if not torch.cuda.is_available():raise RuntimeError("CUDA required")
            adapter=FrozenVideoMAEAdapter();before=encoder_state_hash(adapter.encoder)
            expected=json.loads((PILOT/"pilot_validation.json").read_text(encoding="utf-8"))["encoder_state_before"]
            assert before==expected,"Encoder differs from the passed Pilot"
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            print(json.dumps(dict(stage="frozen_encoder_ready",GPU=torch.cuda.get_device_name(0),state_matches_Pilot=True)),flush=True)
        last_print=time.perf_counter()
        try:
            for split,ds in datasets.items():
                for length in range(1,17):
                    indices=[i for i in pending[split] if ds.records[i]["effective_length"]==length]
                    for offset in range(0,len(indices),2):
                        group=indices[offset:offset+2];tick=time.perf_counter()
                        items=[ds[i] for i in group];decoded+=len(items)
                        pixels=torch.stack([item["frames"] for item in items]).cuda()
                        assert pixels.shape==(len(group),length,3,224,398)
                        output=adapter(pixels);features=output["features"].float().cpu().numpy().copy()
                        for j,index in enumerate(group):
                            r=ds.records[index];item=items[j]
                            assert item["frame_indices"].tolist()==r["frame_indices"] and item["target"].tolist()==r["target_119"]
                            assert item["label_sources"]==r["label_evidence"]
                            commit(r,ctx,np.ascontiguousarray(features[j]),output["geometry"])
                            counts[split]+=1;new+=1
                        torch.cuda.synchronize()
                        batch_times.append(dict(split=split,real_length=length,batch=len(group),seconds=time.perf_counter()-tick))
                        del items,pixels,output,features
                        if new%64<2 or time.perf_counter()-last_print>=30:
                            progress=dict(stage="extracting",split=split,real_length=length,completed=counts,
                                          new_encoder_clips=new,Pilot_copied=copied,reused_full=reused,seconds=time.perf_counter()-started,
                                          CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                          RAM=ram.result())
                            write_json(OUT/"progress.json",progress);print(json.dumps(progress),flush=True);last_print=time.perf_counter()
                    print(json.dumps(dict(stage="length_complete",split=split,real_length=length,completed=counts[split])),flush=True)
            if adapter is not None:
                torch.cuda.synchronize();allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                assert encoder_state_hash(adapter.encoder)==before
                assert not adapter.encoder.training and all(not p.requires_grad and p.grad is None for p in adapter.encoder.parameters())
                adapter.cpu();adapter.position_cache.clear();gc.collect();torch.cuda.empty_cache()
            assert counts=={"train":4667,"valid":979}
        except Exception as e:
            failed.append(dict(error_type=type(e).__name__,error=str(e),split=locals().get("split"),indices=locals().get("group")))
            write_json(run_path,dict(status="interrupted_or_failed",completed=counts,new_extractions=new,Pilot_copied=copied,reused_full=reused,
                                     failures=failed,seconds=time.perf_counter()-started,Test_read=False))
            raise
        result=dict(status="passed",mode="resume_check" if resume_check else "full",counts=counts,new_extractions=new,Pilot_copied=copied,reused_full=reused,
                    decoded_clips=decoded,failed=0,failures=failed,seconds=time.perf_counter()-started,encoder_phase_seconds=time.perf_counter()-extraction_start,
                    CUDA_peak_allocated_bytes=allocated,CUDA_peak_reserved_bytes=reserved,RAM=ram.result(),batch_timings=batch_times,
                    batch_size_max=2,same_real_length_buckets=True,precision="fp32",attention="sdpa",encoder_state_before=before,encoder_state_after=before,
                    encoder_frozen=True,gradient_or_optimizer_steps=0,Test_read=False,new_weights_downloaded=False,formal_training=False)
    write_json(run_path,result)
    target=OUT/("resume_check.json" if resume_check else "extraction_summary.json")
    if target.exists():raise FileExistsError("Keep an already completed summary: "+str(target))
    write_json(target,result);write_json(OUT/"progress.json",dict(status="completed",counts=counts,failed=0))
    print(json.dumps({k:v for k,v in result.items() if k!="batch_timings"}),flush=True)


if __name__=="__main__":main()
