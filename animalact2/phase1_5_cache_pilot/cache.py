"""Uncompressed FP32 frame shards and variable-length clip arrays, with atomic commits."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from animalact2.phase1_5_1.dataset import ReleaseClipDataset, RELEASE_ID
from animalact2.phase1_5_1.backbone import ResNet18Backbone, OFFICIAL_SHA256
from animalact2.phase1_5_1.transforms import FullFrameTransform
from animalact2.phase1_5_1.video_decoder import decode_exact_frames
from animalact2.phase1_5.config import seed_everything
from .support import ROOT, OUT, sha, digest, write_json, RAMMonitor

CACHE=OUT/"cache"


class CacheIntegrityError(ValueError):pass


def context():
    p=ROOT/"animalact2/release_v1/preprocessing.json"
    return dict(schema="animalact2_resnet18_frame_fp32_v1",release_id=RELEASE_ID,
                architecture="torchvision.resnet18.fc_identity",weights_SHA256=OFFICIAL_SHA256,
                preprocessing=json.loads(p.read_text(encoding="utf-8")),preprocessing_file_SHA256=sha(p),
                processing_code_SHA256={name:sha(ROOT/"animalact2/phase1_5_1"/(name+".py")) for name in ("backbone","transforms","video_decoder")},
                torch=str(torch.__version__),torchvision=str(torchvision.__version__),Pillow=pillow_version,
                dtype="float32",feature_dimension=512,TF32=False,frame_chunk_size=8,backbone_mode="frozen_eval")


def frame_descriptor(record,index,ctx):
    return dict(video_id=record["parent_video_id"],split=record["split"],video_SHA256=record["video_sha256"],
                frame_index=int(index),preprocessing_SHA256=digest(ctx["preprocessing"]),backbone_SHA256=ctx["weights_SHA256"],
                context_SHA256=digest(ctx))


def frame_key(record,index,ctx):return digest(frame_descriptor(record,index,ctx))


def label_metadata(record):
    return {k:record[k] for k in ("clip_id","split","parent_video_id","original_clip_ids","frame_indices","effective_length","target_119","class_ids","source_annotation_uids","label_evidence","annotation_sha256")}


def clip_descriptor(record,ctx):
    return dict(clip_id=record["clip_id"],split=record["split"],context_SHA256=digest(ctx),
                frame_keys=[frame_key(record,k,ctx) for k in record["frame_indices"]],label_metadata_SHA256=digest(label_metadata(record)),
                shape=[record["effective_length"],512],dtype="float32")


def select_plan():
    path=OUT/"selection.json";ctx=context()
    if path.exists():
        plan=json.loads(path.read_text(encoding="utf-8"))
        if plan["context"]!=ctx:raise CacheIntegrityError("Preprocessing/backbone context changed")
        return plan
    full={s:ReleaseClipDataset(s) for s in ("train","valid")};rng=np.random.default_rng(42)
    selections={}
    forced_pair=None
    for split,quotas in (("train",(3,20,77)),("valid",(1,3,16))):
        records=full[split].records;chosen=[]
        for n in ([1,2,3,4,15] if split=="train" else [1,4,10,15]):
            candidates=[r for r in records if r["effective_length"]==n and r["clip_id"] not in chosen]
            chosen.append(candidates[int(rng.integers(len(candidates)))]["clip_id"])
        if split=="train":
            grouped=defaultdict(list)
            for r in records:
                if r["effective_length"]==16:grouped[r["parent_video_id"]].append(r)
            for vid,rr in sorted(grouped.items()):
                pairs=[(a,b) for i,a in enumerate(rr) for b in rr[i+1:] if set(a["frame_indices"])&set(b["frame_indices"])]
                if pairs:
                    a,b=pairs[0];chosen.extend([a["clip_id"],b["clip_id"]]);forced_pair=[a["clip_id"],b["clip_id"]];break
        for (lo,hi),quota in zip(((1,3),(4,15),(16,16)),quotas):
            existing=sum(lo<=r["effective_length"]<=hi and r["clip_id"] in chosen for r in records)
            candidates=[r["clip_id"] for r in records if lo<=r["effective_length"]<=hi and r["clip_id"] not in chosen]
            chosen.extend(rng.choice(candidates,size=quota-existing,replace=False).tolist())
        assert len(chosen)==len(set(chosen))==sum(quotas)
        rng.shuffle(chosen);selections[split]=chosen
    selected=[r for s,d in full.items() for cid in selections[s] for r in d.records if r["clip_id"]==cid]
    source_stats={}
    for r in selected:
        p=ROOT/r["video_path"];st=p.stat()
        source_stats[r["parent_video_id"]]=dict(path=r["video_path"],bytes=st.st_size,mtime_ns=st.st_mtime_ns,SHA256=r["video_sha256"])
    all_records=[r for d in full.values() for r in d.records]
    all_unique={(r["parent_video_id"],r["video_sha256"],k) for r in all_records for k in r["frame_indices"]}
    plan=dict(scope="only 100 Train and 20 Valid clips",release_id=RELEASE_ID,seed=42,context=ctx,selections=selections,source_stats=source_stats,
              quotas=dict(train=[3,20,77],valid=[1,3,16]),forced_overlapping_frame_pair=forced_pair,
              full_Train_Valid_metadata_only=dict(clips=len(all_records),unique_frames=len(all_unique),clip_frame_references=sum(r["effective_length"] for r in all_records)),
              manifest_SHA256={s:full[s].manifest_sha256 for s in full},Test_opened=False)
    write_json(path,plan)
    with (OUT/"selected_manifest.jsonl").open("w",encoding="utf-8") as f:
        for r in selected:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
    return plan


def selected_records(plan):
    if {s:len(v) for s,v in plan["selections"].items()}!={"train":100,"valid":20}:raise ValueError("Pilot cap is exactly 120 clips")
    rr=[]
    for split in ("train","valid"):
        source=ReleaseClipDataset(split,clip_ids=plan["selections"][split])
        if source.manifest_sha256!=plan["manifest_SHA256"][split]:raise CacheIntegrityError("Manifest changed")
        rr.extend(source.records)
    return rr


def commit_array(directory,array,receipt):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(OUT):raise ValueError("Cache output outside pilot")
    if directory.exists():raise FileExistsError("Do not overwrite a committed cache entry")
    if array.dtype!=np.float32 or not np.isfinite(array).all():raise CacheIntegrityError("Expected finite FP32 features")
    staging=directory.parent/(".pending_"+uuid.uuid4().hex);staging.mkdir(parents=True,exist_ok=False)
    with (staging/"features.npy").open("wb") as f:
        np.save(f,np.ascontiguousarray(array),allow_pickle=False);f.flush();os.fsync(f.fileno())
    receipt=dict(receipt,array_SHA256=sha(staging/"features.npy"),array_bytes=(staging/"features.npy").stat().st_size,
                 array_shape=list(array.shape),array_dtype="float32",compression="none")
    write_json(staging/"receipt.json",receipt)
    # Atomic directory rename commits both feature array and provenance together.
    staging.rename(directory)


def read_array(directory,expected=None):
    directory=Path(directory);receipt=json.loads((directory/"receipt.json").read_text(encoding="utf-8"))
    if expected and any(receipt.get(k)!=v for k,v in expected.items()):raise CacheIntegrityError("Cache metadata/key mismatch: "+str(directory))
    p=directory/"features.npy"
    if p.stat().st_size!=receipt["array_bytes"] or sha(p)!=receipt["array_SHA256"]:raise CacheIntegrityError("Feature SHA/size mismatch: "+str(p))
    with p.open("rb") as f:array=np.load(f,allow_pickle=False)
    if array.dtype!=np.float32 or list(array.shape)!=receipt["array_shape"] or not np.isfinite(array).all():raise CacheIntegrityError("Invalid feature shape/dtype/values")
    return array,receipt


class FrameStore:
    def __init__(self,root,ctx):
        self.root=Path(root);self.ctx=ctx;self.context_hash=digest(ctx);self.locations={}
        context_path=self.root/"context.json"
        if context_path.exists():
            if json.loads(context_path.read_text(encoding="utf-8"))!=ctx:raise CacheIntegrityError("Wrong preprocessing or weights")
        else:write_json(context_path,ctx)
        for directory in sorted((self.root/"frames").glob("shard_*")):
            array,receipt=read_array(directory,dict(context_SHA256=self.context_hash,kind="unique_frame_shard"))
            entries=receipt["frames"]
            if array.shape!=(len(entries),512):raise CacheIntegrityError("Wrong frame shard shape")
            for row,entry in enumerate(entries):
                key=digest(entry)
                if key in self.locations:raise CacheIntegrityError("Duplicate committed unique frame")
                if entry["context_SHA256"]!=self.context_hash:raise CacheIntegrityError("Frame context mismatch")
                self.locations[key]=(directory,row)

    def commit(self,descriptors,array):
        keys=[digest(d) for d in descriptors]
        if len(set(keys))!=len(keys) or any(k in self.locations for k in keys):raise CacheIntegrityError("Attempted duplicate extraction/commit")
        directory=self.root/"frames"/("shard_"+digest(keys))
        commit_array(directory,array,dict(context_SHA256=self.context_hash,kind="unique_frame_shard",frames=descriptors))
        for row,key in enumerate(keys):self.locations[key]=(directory,row)

    @lru_cache(maxsize=32)
    def _array(self,path):return read_array(path,dict(context_SHA256=self.context_hash,kind="unique_frame_shard"))[0]

    def get(self,key):
        directory,row=self.locations[key];return self._array(str(directory))[row]


def build_cache(*,max_new_frames=None):
    begin=time.perf_counter();seed_everything(42);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    plan=select_plan();ctx=plan["context"];records=selected_records(plan);store=FrameStore(CACHE,ctx)
    groups=defaultdict(list)
    for r in records:groups[(r["split"],r["parent_video_id"])].append(r)
    expected={frame_key(r,k,ctx) for r in records for k in r["frame_indices"]}
    if not set(store.locations)<=expected:raise CacheIntegrityError("Frame bank includes clips outside this locked pilot")
    reused=len(store.locations);new=0;decoded_parents=0;encoder=None;stopped=False
    with RAMMonitor() as ram:
        for (split,vid),rr in sorted(groups.items()):
            first=rr[0];indices=sorted({k for r in rr for k in r["frame_indices"] if frame_key(r,k,ctx) not in store.locations})
            if not indices:continue
            if encoder is None:encoder=ResNet18Backbone(frozen=True,frame_chunk_size=8).to("cuda").eval()
            p=ROOT/first["video_path"]
            if sha(p)!=first["video_sha256"]:raise CacheIntegrityError("Source video SHA changed")
            stat=p.stat();saved=plan["source_stats"][vid]
            if (stat.st_size,stat.st_mtime_ns)!=(saved["bytes"],saved["mtime_ns"]):raise CacheIntegrityError("Source video stat changed")
            decoded=decode_exact_frames(p,indices,expected_num_frames=first["parent_num_frames"],expected_fps=first["fps"]);decoded_parents+=1
            transform=FullFrameTransform()
            for start in range(0,len(indices),8):
                if max_new_frames is not None and new>=max_new_frames:stopped=True;break
                chosen=indices[start:start+8]
                frames=torch.stack([transform(rgb) for rgb in decoded.rgb[start:start+8]]).to("cuda")
                with torch.inference_mode():features=encoder.encoder(frames).float().cpu().numpy().copy()
                assert not encoder.encoder.training and not any(p.requires_grad for p in encoder.parameters())
                descriptors=[frame_descriptor(first,k,ctx) for k in chosen]
                store.commit(descriptors,features);new+=len(chosen)
            if stopped:break
        complete=set(store.locations)==expected;new_clips=0;reused_clips=0
        if complete:
            for r in records:
                descriptor=clip_descriptor(r,ctx);directory=CACHE/"clips"/r["split"]/r["clip_id"]
                if directory.exists():read_array(directory,dict(clip_descriptor=descriptor,clip_key=digest(descriptor),kind="clip_features"));reused_clips+=1
                else:
                    matrix=np.stack([store.get(k) for k in descriptor["frame_keys"]]).astype(np.float32,copy=False)
                    commit_array(directory,matrix,dict(clip_descriptor=descriptor,clip_key=digest(descriptor),kind="clip_features"));new_clips+=1
            # Audit exact matrix rows against the shared frame bank, labels remain in release-v1.
            for r in records:
                a,_=read_array(CACHE/"clips"/r["split"]/r["clip_id"],dict(clip_descriptor=clip_descriptor(r,ctx)))
                if not np.array_equal(a,np.stack([store.get(frame_key(r,k,ctx)) for k in r["frame_indices"]])):raise CacheIntegrityError("Clip/frame alignment mismatch")
    torch.cuda.synchronize()
    result=dict(status="complete" if complete else "deliberately_interrupted",unique_frames_expected=len(expected),new_unique_frames_inferred=new,reused_valid_unique_frames=reused,
                CNN_frames_evaluated_for_cache_build=new,decoded_parent_videos=decoded_parents,new_clips=new_clips,reused_clips=reused_clips,
                elapsed_seconds=time.perf_counter()-begin,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),RAM=ram.result(),
                context_SHA256=digest(ctx),FP32_uncompressed=True,Test_read=False,full_cache_created=False)
    runs=OUT/"extraction_runs";runs.mkdir(exist_ok=True);index=len(list(runs.glob("run_*.json")))+1
    write_json(runs/f"run_{index:02d}.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result
