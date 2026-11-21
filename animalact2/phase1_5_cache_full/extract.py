"""Generate exactly release-v1 Train/Valid features; never train or load task checkpoints."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import shutil
import time
import traceback
import cv2
import numpy as np
import torch
from animalact2.phase1_5_1.dataset import ReleaseClipDataset,RELEASE_ID
from animalact2.phase1_5_1.backbone import ResNet18Backbone
from animalact2.phase1_5_1.transforms import FullFrameTransform
from animalact2.phase1_5_1.video_decoder import decode_exact_frames
from animalact2.phase1_5.config import seed_everything
from .storage import CACHE,FrameStore,context,frame_descriptor,frame_key,clip_descriptor,read_array,commit_array,CacheIntegrityError
from .support import ROOT,OUT,PILOT,enable_guard,sha,digest,write_json,RAMMonitor


def load_plan():
    ctx=context();pilot_context=json.loads((PILOT/"cache/context.json").read_text(encoding="utf-8"))
    if ctx!=pilot_context:raise CacheIntegrityError("Must use exactly the approved Pilot numerical/preprocessing/weight context")
    source={s:ReleaseClipDataset(s) for s in ("train","valid")}
    assert {s:len(d) for s,d in source.items()}==dict(train=4667,valid=979)
    records=[r for d in source.values() for r in d.records]
    unique={frame_key(r,k,ctx) for r in records for k in r["frame_indices"]}
    assert len(unique)==73884 and sum(r["effective_length"] for r in records)==81641
    path=OUT/"full_plan.json"
    if path.exists():
        plan=json.loads(path.read_text(encoding="utf-8"))
        if plan["context"]!=ctx or plan["manifest_SHA256"]!={s:d.manifest_sha256 for s,d in source.items()}:raise CacheIntegrityError("Context or Manifest changed")
        for r in records:
            saved=plan["source_stats"][r["parent_video_id"]];st=(ROOT/r["video_path"]).stat()
            if (st.st_size,st.st_mtime_ns)!=(saved["bytes"],saved["mtime_ns"]):raise CacheIntegrityError("Source video changed")
        return plan,records
    if shutil.disk_usage(OUT).free<2*1024**3:raise RuntimeError("Need at least2GiB free for independent FP32 cache and validation")
    stats={}
    for r in records:
        vid=r["parent_video_id"]
        if vid in stats:continue
        p=ROOT/r["video_path"];a=ROOT/r["annotation_path"];st=p.stat()
        if sha(p)!=r["video_sha256"] or sha(a)!=r["annotation_sha256"]:raise CacheIntegrityError("Source identity hash mismatch: "+vid)
        stats[vid]=dict(split=r["split"],path=r["video_path"],bytes=st.st_size,mtime_ns=st.st_mtime_ns,SHA256=r["video_sha256"],
                        annotation_path=r["annotation_path"],annotation_SHA256=r["annotation_sha256"],FPS=r["fps"],parent_num_frames=r["parent_num_frames"])
    plan=dict(scope="complete fixed Train/Valid only",release_id=RELEASE_ID,context=ctx,selections={s:[r["clip_id"] for r in d.records] for s,d in source.items()},
              source_stats=stats,manifest_SHA256={s:d.manifest_sha256 for s,d in source.items()},unique_frames=73884,clip_frame_references=81641,
              source_video_and_annotation_SHA_verified=True,Test_opened=False,created_unix=time.time(),Pilot_reuse="SHA-validated independent copies; no links or writes to Pilot")
    write_json(path,plan);return plan,records


def seed_from_pilot(store,expected_descriptors):
    added=0;copied_shards=0;pilot_keys=set()
    for directory in sorted((PILOT/"cache/frames").glob("shard_*")):
        array,receipt=read_array(directory,dict(context_SHA256=store.context_hash,kind="unique_frame_shard"))
        descriptors=receipt["frames"]
        if array.shape!=(len(descriptors),512):raise CacheIntegrityError("Invalid Pilot shard shape")
        rows=[];missing=[]
        for row,entry in enumerate(descriptors):
            key=digest(entry)
            if key not in expected_descriptors or entry!=expected_descriptors[key]:raise CacheIntegrityError("Pilot frame identity mismatch")
            if key in pilot_keys:raise CacheIntegrityError("Duplicate Pilot frame")
            pilot_keys.add(key)
            if key not in store.locations:rows.append(row);missing.append(entry)
        if missing:
            store.commit(missing,np.ascontiguousarray(array[rows]));added+=len(rows);copied_shards+=1
    assert len(pilot_keys)==1699
    return dict(Pilot_valid_unique_frames=1699,independent_copied_frames_this_run=added,copied_shards_this_run=copied_shards,linking=False)


def run(*,pause_after=None):
    begin=time.perf_counter();seed_everything(42);torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    runs=OUT/"extraction_runs";runs.mkdir(exist_ok=True);run_id=len(list(runs.glob("run_*.json")))+1
    errors=[];write_json(OUT/"build_errors.json",errors)
    new=0;decoded=0;new_clips=0;reused_clips=0;status="interrupted";seed_info={};existing=0
    with RAMMonitor() as ram:
        try:
            plan,records=load_plan();ctx=plan["context"]
            descriptors={frame_key(r,k,ctx):frame_descriptor(r,k,ctx) for r in records for k in r["frame_indices"]}
            store=FrameStore(CACHE,ctx);existing=len(store.locations)
            if not set(store.locations)<=set(descriptors):raise CacheIntegrityError("Full bank has frames outside approved Train/Valid")
            seed_info=seed_from_pilot(store,descriptors)
            groups=defaultdict(list)
            for r in records:groups[(r["split"],r["parent_video_id"])].append(r)
            encoder=None;transform=FullFrameTransform();paused=False;done=0
            for (split,vid),rr in sorted(groups.items()):
                first=rr[0];indices=sorted({k for r in rr for k in r["frame_indices"] if frame_key(r,k,ctx) not in store.locations})
                if indices:
                    if encoder is None:encoder=ResNet18Backbone(frozen=True,frame_chunk_size=8).to("cuda").eval()
                    decoded_video=decode_exact_frames(ROOT/first["video_path"],indices,expected_num_frames=first["parent_num_frames"],expected_fps=first["fps"]);decoded+=1
                    for start in range(0,len(indices),8):
                        if pause_after is not None and new>=pause_after:paused=True;break
                        chosen=indices[start:start+8]
                        images=torch.stack([transform(rgb) for rgb in decoded_video.rgb[start:start+8]]).to("cuda")
                        with torch.inference_mode():features=encoder.encoder(images).float().cpu().numpy().copy()
                        if features.shape!=(len(chosen),512) or features.dtype!=np.float32 or not np.isfinite(features).all():raise CacheIntegrityError("Invalid frozen ResNet18 output")
                        assert not encoder.encoder.training and not any(p.requires_grad for p in encoder.parameters())
                        store.commit([frame_descriptor(first,k,ctx) for k in chosen],features);new+=len(chosen)
                    del decoded_video
                    if paused:break
                done+=1
                if done%100==0 or done==len(groups):
                    progress=dict(phase="frame_extraction",parents_done=done,total_parents=len(groups),available_unique_frames=len(store.locations),expected=73884,
                                  new_CNN_frames=new,copied_Pilot_frames=seed_info["independent_copied_frames_this_run"],seconds=round(time.perf_counter()-begin,2))
                    write_json(OUT/"build_progress.json",progress);print(json.dumps(progress),flush=True)
            if not paused and set(store.locations)==set(descriptors):
                for i,r in enumerate(records,1):
                    desc=clip_descriptor(r,ctx);directory=CACHE/"clips"/r["split"]/r["clip_id"]
                    if directory.exists():
                        read_array(directory,dict(clip_descriptor=desc,clip_key=digest(desc),kind="clip_features"));reused_clips+=1
                    else:
                        matrix=np.stack([store.get(key) for key in desc["frame_keys"]]).astype(np.float32,copy=False)
                        commit_array(directory,matrix,dict(clip_descriptor=desc,clip_key=digest(desc),kind="clip_features"));new_clips+=1
                    if i%500==0 or i==len(records):
                        progress=dict(phase="clip_assembly",clips_done=i,total_clips=len(records),new_clips=new_clips,reused_clips=reused_clips,seconds=round(time.perf_counter()-begin,2))
                        write_json(OUT/"build_progress.json",progress);print(json.dumps(progress),flush=True)
                status="complete"
            else:status="paused_at_shard_boundary"
        except BaseException as exc:
            errors.append(dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
            write_json(OUT/"build_errors.json",errors);status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
            raise
        finally:
            torch.cuda.synchronize()
            result=dict(status=status,run_id=run_id,existing_valid_unique_frames=existing,Pilot_reuse=seed_info,new_CNN_unique_frames=new,
                        committed_unique_frames=len(store.locations) if "store" in locals() else None,expected_unique_frames=73884,
                        decoded_parent_videos=decoded,new_clips=new_clips,reused_clips=reused_clips,expected_Train_clips=4667,expected_Valid_clips=979,
                        elapsed_seconds=time.perf_counter()-begin,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        RAM=ram.result(),failure_count=len(errors),FP32_uncompressed=True,backbone_frozen_eval_inference_mode=True,
                        context_SHA256=digest(plan["context"]) if "plan" in locals() else None,Test_read=False,training_started=False,Checkpoint_modified=False)
            write_json(runs/f"run_{run_id:02d}.json",result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    enable_guard();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause-after-new-frames",type=int,help="Optional controlled pause at a complete shard boundary, max7 frames beyond threshold")
    args=parser.parse_args()
    if args.pause_after_new_frames is not None and args.pause_after_new_frames<1:raise ValueError("Positive pause threshold required")
    run(pause_after=args.pause_after_new_frames)


if __name__=="__main__":main()
