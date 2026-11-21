"""Fixed 100/20 selection, immutable Clip provenance and resumable FP32 cache."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from .support import ROOT,OUT,RELEASE,SOURCE,sha,digest,write_json

FORCED={"train":{
    "aa2v1_8a16e630c2194b57d1e086b2":"reviewed_left_lower_edge_cricket",
    "aa2v1_989dab5adaf16998cc564c45":"single_frame_small_bird_multilabel",
    "aa2v1_3c345e3b906d381efd73f1fe":"small_birds_full_width",
    "aa2v1_5ca2c42a07988af2e3473f5f":"reviewed_multiple_horizontal_waterbirds",
    "aa2v1_9408cc971fd41195e5a136ed":"reviewed_multiple_goats",
    "aa2v1_811be0c0daf59d5466624157":"reviewed_full_body_dog",
    "aa2v1_e0c80382800b865461f68192":"entire_source_only_three_real_frames"},
    "valid":{"aa2v1_31b73955473fd8b6d430a525":"reviewed_multiple_waterfowl"}}


def get_plan():
    path=OUT/"pilot_plan.json"
    if path.exists():
        p=json.loads(path.read_text(encoding="utf-8"))
        assert p["manifest_SHA256"]=={s:sha(RELEASE/s/"manifest.jsonl") for s in ("train","valid")}
        for split,count in (("train",100),("valid",20)):
            original={r["clip_id"]:r for r in ReleaseClipDataset(split).records}
            assert len(p["records"][split])==count and len({r["clip_id"] for r in p["records"][split]})==count
            assert all(r==original[r["clip_id"]] for r in p["records"][split])
        return p
    plan=dict(seed=42,selection="purposeful technical coverage, not a statistical performance sample",manifest_SHA256={},records={},reasons={})
    for split in ("train","valid"):
        ds=ReleaseClipDataset(split)
        quota={n:4 for n in range(1,17)} if split=="train" else {n:1 for n in range(1,17)}
        if split=="train":quota.update({3:8,15:8,16:32})
        else:quota.update({3:2,15:2,16:3})
        assert sum(quota.values())==(100 if split=="train" else 20)
        by_id={r["clip_id"]:r for r in ds.records};chosen=set(FORCED[split]);assert chosen<=by_id.keys()
        for n,count in quota.items():
            already=sum(by_id[c]["effective_length"]==n for c in chosen)
            assert already<=count
            candidates=[r for r in ds.records if r["effective_length"]==n and r["clip_id"] not in chosen]
            candidates.sort(key=lambda r:(not bool(r["sampled_label_coverage_warnings"]),-min(len(r["class_ids"]),6),digest([42,r["clip_id"]])))
            chosen.update(r["clip_id"] for r in candidates[:count-already])
        plan["records"][split]=[r for r in ds.records if r["clip_id"] in chosen]
        assert len(plan["records"][split])==sum(quota.values())
        plan["manifest_SHA256"][split]=ds.manifest_sha256
        plan["reasons"][split]={r["clip_id"]:FORCED[split].get(r["clip_id"],"length_stratification_multilabel_and_sampling_warning_priority") for r in plan["records"][split]}
    assert len(set(r["clip_id"] for rows in plan["records"].values() for r in rows))==120
    write_json(path,plan)
    rows=[]
    for split,records in plan["records"].items():
        for r in records:
            rows.append(dict(split=split,clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],real_length=r["effective_length"],
                             tubelet_length=(r["effective_length"]+1)//2,frame_indices=json.dumps(r["frame_indices"]),
                             target_119=json.dumps(r["target_119"]),class_ids=json.dumps(r["class_ids"]),
                             start=r["effective_start_exact"],end=r["effective_end_exact"],
                             source_annotation_uids=json.dumps(r["source_annotation_uids"]),selection_reason=plan["reasons"][split][r["clip_id"]]))
    with (OUT/"sample_manifest.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return plan


def cache_context():
    source=json.loads(SOURCE.read_text(encoding="utf-8"))
    from importlib.metadata import version
    return dict(schema="animalact2_videomae_rect_variable_fp32_pilot_v1",release="animalact2-2.0.0-release-v1",source=source,
                preprocessing="release RGB PIL bilinear short224, full frame 224x398, ImageNet mean/std; no crop/warp/context/repeat",
                alignment="normalized zeros: right2 pixels; one trailing temporal slot only for odd real L; no padding to16",
                position="native sincos [8,14,14,768]; spatial bicubic, temporal linear, align_corners=False; native bypass exact",
                extraction="same real-length buckets, frozen eval inference_mode, CUDA FP32 SDPA, no random token masking",
                pooling="mean spatial grid; retain every ceil(L/2) step including partially real tubelet",
                feature_dtype="float32",transformers=version("transformers"),torch=str(torch.__version__),
                implementation_SHA256={n:sha(OUT/n) for n in ("adapter.py","data.py")})


def descriptor(r,ctx):
    return dict(context_SHA256=digest(ctx),clip_id=r["clip_id"],split=r["split"],parent_video_id=r["parent_video_id"],
                video_path=r["video_path"],video_content_SHA256=sha(ROOT/r["video_path"]),frame_indices=r["frame_indices"],
                real_length=r["effective_length"],effective_start=r["effective_start_exact"],effective_end=r["effective_end_exact"],
                label_and_provenance_SHA256=digest(dict(target=r["target_119"],evidence=r["label_evidence"])))


def read_cache(r,ctx):
    d=OUT/"cache"/r["split"]/r["clip_id"]
    if not d.exists():return None
    receipt=json.loads((d/"receipt.json").read_text(encoding="utf-8"));desc=descriptor(r,ctx)
    assert receipt["descriptor"]==desc and receipt["key"]==digest(desc),"Cache identity mismatch"
    assert sha(d/"features.npy")==receipt["features_SHA256"],"Cache content changed"
    x=np.load(d/"features.npy",allow_pickle=False)
    assert x.dtype==np.float32 and x.shape==((r["effective_length"]+1)//2,768) and np.isfinite(x).all()
    assert receipt["frame_indices"]==r["frame_indices"] and receipt["target_119"]==r["target_119"]
    assert receipt["label_evidence"]==r["label_evidence"]
    flatten=[v for pair in receipt["tubelet_frame_indices"] for v in pair if v is not None]
    assert flatten==r["frame_indices"] and len(flatten)==len(set(flatten))
    assert receipt["valid_tubelet_mask"]==[True]*len(x)
    assert receipt["tubelet_real_counts"]==[min(2,r["effective_length"]-i) for i in range(0,r["effective_length"],2)]
    return x,receipt


def commit(r,ctx,features,geometry):
    desc=descriptor(r,ctx);d=OUT/"cache"/r["split"]/r["clip_id"]
    if d.exists():raise FileExistsError("Never overwrite a committed pilot cache")
    assert features.dtype==np.float32 and features.shape==((r["effective_length"]+1)//2,768) and np.isfinite(features).all()
    pending=d.parent/f".pending_{r['clip_id']}_{time.time_ns()}";pending.mkdir(parents=True)
    with (pending/"features.npy").open("wb") as f:
        np.save(f,features,allow_pickle=False);f.flush();os.fsync(f.fileno())
    indices=r["frame_indices"]
    receipt=dict(descriptor=desc,key=digest(desc),features_SHA256=sha(pending/"features.npy"),feature_shape=list(features.shape),dtype="float32",
                 clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],split=r["split"],frame_indices=indices,target_119=r["target_119"],
                 tubelet_frame_indices=[indices[i:i+2]+([None] if i+1==len(indices) else []) for i in range(0,len(indices),2)],
                 tubelet_real_counts=geometry["tubelet_real_counts"],valid_tubelet_mask=[True]*len(features),geometry=geometry,
                 label_evidence=r["label_evidence"])
    with (pending/"receipt.json").open("w",encoding="utf-8") as f:
        f.write(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False));f.flush();os.fsync(f.fileno())
    pending.rename(d)
    return read_cache(r,ctx)


class CachedPilotDataset(Dataset):
    def __init__(self,split):
        if split not in ("train","valid"):raise ValueError("Pilot permits Train/Valid only")
        self.split=split;self.plan=get_plan();self.records=self.plan["records"][split];self.ctx=cache_context()
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i];item=read_cache(r,self.ctx)
        if item is None:raise FileNotFoundError(r["clip_id"])
        x,receipt=item
        return dict(features=torch.from_numpy(x.copy()),length=len(x),real_length=r["effective_length"],target=torch.tensor(r["target_119"],dtype=torch.float32),
                    clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],split=self.split,frame_indices=r["frame_indices"],metadata=r)


def collate_features(samples):
    lengths=torch.tensor([s["length"] for s in samples],dtype=torch.int64)
    x=torch.zeros(len(samples),int(lengths.max()),768,dtype=torch.float32)
    mask=torch.arange(x.shape[1])[None,:]<lengths[:,None]
    for i,s in enumerate(samples):x[i,:s["length"]]=s["features"]
    return dict(features=x,lengths=lengths,valid_frame_mask=mask,padding_mask=~mask,target=torch.stack([s["target"] for s in samples]),
                clip_ids=[s["clip_id"] for s in samples],real_lengths=[s["real_length"] for s in samples],splits=[s["split"] for s in samples])
