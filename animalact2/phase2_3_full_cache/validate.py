"""Independent full-array/receipt/Manifest audit; no decoding or model inference."""
from __future__ import annotations
import collections
import csv
import itertools
import json
import time
import numpy as np
import torch
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from .support import ROOT,OUT,PILOT,RELEASE,enable_guard,sha,digest,write_json
from .storage import CACHE,cache_context
from .plan import verify_sources


def csv_write(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    enable_guard();started=time.perf_counter()
    ctx=json.loads((CACHE/"context.json").read_text(encoding="utf-8"))
    assert ctx==cache_context()==json.loads((PILOT/"cache/context.json").read_text(encoding="utf-8"))
    plan=json.loads((OUT/"full_plan.json").read_text(encoding="utf-8"))
    assert ctx==plan["context"] and not plan["Test_enabled"]
    datasets={s:ReleaseClipDataset(s) for s in ("train","valid")}
    sources=verify_sources(datasets)
    pilot_plan=json.loads((PILOT/"pilot_plan.json").read_text(encoding="utf-8"))
    pilot_ids={r["clip_id"] for rows in pilot_plan["records"].values() for r in rows}
    assert len(pilot_ids)==120
    checks=[];split_stats=[];class_counts={};parent_sets={};copied_checked=0;keys=set();errors=[]
    real_refs=time_steps=positive_pairs=contributor_links=0
    for split,ds in datasets.items():
        assert ds.manifest_sha256==plan["manifest_SHA256"][split]
        assert [r["clip_id"] for r in ds.records]==plan["selections"][split]
        expected_ids={r["clip_id"] for r in ds.records}
        assert {p.name for p in (CACHE/split).iterdir() if not p.name.startswith(".pending_")}==expected_ids
        counts=np.zeros(119,dtype=np.int64);parents=[set() for _ in range(119)];feature_min=float("inf");feature_max=float("-inf")
        for r in ds.records:
            d=CACHE/split/r["clip_id"];a=np.load(d/"features.npy",allow_pickle=False)
            receipt=json.loads((d/"receipt.json").read_text(encoding="utf-8"))
            l=r["effective_length"];lt=(l+1)//2
            assert a.dtype==np.float32 and a.shape==(lt,768) and a.flags.c_contiguous and np.isfinite(a).all()
            assert receipt["features_SHA256"]==sha(d/"features.npy")
            expected_desc=dict(context_SHA256=digest(ctx),clip_id=r["clip_id"],split=split,parent_video_id=r["parent_video_id"],
                               video_path=r["video_path"],video_content_SHA256=sources[r["video_path"]]["sha256"],frame_indices=r["frame_indices"],
                               real_length=l,effective_start=r["effective_start_exact"],effective_end=r["effective_end_exact"],
                               label_and_provenance_SHA256=digest(dict(target=r["target_119"],evidence=r["label_evidence"])))
            assert receipt["descriptor"]==expected_desc and receipt["key"]==digest(expected_desc)
            assert receipt["key"] not in keys;keys.add(receipt["key"])
            assert receipt["clip_id"]==r["clip_id"] and receipt["parent_video_id"]==r["parent_video_id"] and receipt["split"]==split
            assert receipt["dtype"]=="float32" and receipt["feature_shape"]==[lt,768]
            assert receipt["target_119"]==r["target_119"] and receipt["label_evidence"]==r["label_evidence"]
            assert receipt["frame_indices"]==r["frame_indices"]
            it=iter(r["frame_indices"]);pairs=[list(p) for p in itertools.zip_longest(it,it,fillvalue=None)]
            assert receipt["tubelet_frame_indices"]==pairs and [i for p in pairs for i in p if i is not None]==r["frame_indices"]
            real_counts=[sum(i is not None for i in p) for p in pairs]
            assert receipt["tubelet_real_counts"]==real_counts and sum(real_counts)==l and min(real_counts)>=1
            assert receipt["valid_tubelet_mask"]==[True]*lt
            expected_geometry=dict(real_length=l,aligned_length=l+l%2,temporal_zero_slots=l%2,real_hw=[224,398],aligned_hw=[224,400],
                                   spatial_bottom_right_zero=[0,2],grid=[lt,14,25],tubelet_real_counts=real_counts)
            assert receipt["geometry"]==expected_geometry
            if r["clip_id"] in pilot_ids:
                old=PILOT/"cache"/split/r["clip_id"]
                assert np.array_equal(a,np.load(old/"features.npy",allow_pickle=False))
                assert receipt==json.loads((old/"receipt.json").read_text(encoding="utf-8"))
                assert sha(d/"features.npy")==sha(old/"features.npy")
                assert not (d/"features.npy").is_symlink()
                copied_checked+=1
            counts+=np.array(r["target_119"],dtype=np.int64)
            for i in r["class_ids"]:parents[i].add(r["parent_video_id"])
            real_refs+=l;time_steps+=lt;positive_pairs+=len(r["class_ids"])
            contributor_links+=sum(len(e["contributors"]) for e in r["label_evidence"])
            feature_min=min(feature_min,float(a.min()));feature_max=max(feature_max,float(a.max()))
            checks.append(dict(split=split,clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],real_length=l,feature_length=lt,
                               feature_shape=json.dumps([lt,768]),source_and_labels_aligned=True,all_real_frames_retained=True,Mask_valid=True,
                               dtype="float32",finite=True,SHA256=receipt["features_SHA256"],Pilot_bitwise_equal=r["clip_id"] in pilot_ids))
        class_counts[split]=counts;parent_sets[split]=parents
        split_stats.append(dict(split=split,clips=len(ds),parent_videos=len({r["parent_video_id"] for r in ds.records}),
                                real_frame_references=sum(r["effective_length"] for r in ds.records),
                                feature_time_steps=sum((r["effective_length"]+1)//2 for r in ds.records),
                                length1_3=sum(r["effective_length"]<=3 for r in ds.records),
                                length4_15=sum(4<=r["effective_length"]<=15 for r in ds.records),length16=sum(r["effective_length"]==16 for r in ds.records),
                                odd_lengths=sum(r["effective_length"]%2 for r in ds.records),
                                single_frames=sum(r["effective_length"]==1 for r in ds.records),feature_min=feature_min,feature_max=feature_max,failed=0))
    assert (len(checks),len(keys),real_refs,time_steps,copied_checked)==(5646,5646,81641,41156,120)
    assert set(np.flatnonzero(class_counts["train"]==0))=={23}
    mapping=json.loads((RELEASE/"class_mapping.json").read_text(encoding="utf-8"))["classes"]
    main_ids=json.loads((RELEASE/"main_evaluation_classes.json").read_text(encoding="utf-8"))["class_ids"]
    class_rows=[dict(class_id=i,name=r["name"],main_evaluation=i in main_ids,Train_positive_clips=int(class_counts["train"][i]),
                     Valid_positive_clips=int(class_counts["valid"][i]),Train_positive_parents=len(parent_sets["train"][i]),Valid_positive_parents=len(parent_sets["valid"][i])) for i,r in enumerate(mapping)]
    assert all(row["Train_positive_clips"]>0 and row["Valid_positive_clips"]>0 for row in class_rows if row["main_evaluation"])
    csv_write(OUT/"clip_validation.csv",checks);csv_write(OUT/"split_statistics.csv",split_stats)
    csv_write(OUT/"class_support_119.csv",class_rows);csv_write(OUT/"main31_support.csv",[r for r in class_rows if r["main_evaluation"]])
    write_json(OUT/"errors.json",errors)
    pending=list(CACHE.rglob(".pending_*"))
    assert not pending,"Uncommitted directories require explicit inspection; not counted as valid cache"
    result=dict(status="passed",verified_clips=5646,verified_keys=5646,real_frame_references=real_refs,feature_time_steps=time_steps,
                Pilot_copied_bitwise_equal=copied_checked,source_files_checked=len(sources),positive_clip_class_pairs=positive_pairs,
                original_annotation_contributor_links=contributor_links,split_statistics=split_stats,failed=0,
                no_uncommitted_directories=True,seconds=time.perf_counter()-started,model_forward_calls=0,video_decodes=0,Test_read=False)
    write_json(OUT/"validation_summary.json",result)
    # Read-only interface check on real cached lengths; no model or optimizer constructed.
    from .dataset import CachedFeatureDataset,collate_features
    interfaces=[]
    for split in ("train","valid"):
        ds=CachedFeatureDataset(split)
        samples=[ds[next(i for i,r in enumerate(ds.records) if r["effective_length"]==n)] for n in (1,3,16)]
        b=collate_features(samples)
        assert b["features"].shape==(3,8,768) and b["target"].shape==(3,119) and b["lengths"].tolist()==[1,2,8]
        assert torch.equal(b["valid_frame_mask"],~b["padding_mask"]) and torch.all(b["features"][b["padding_mask"]]==0)
        assert all(torch.equal(b["features"][i,:s["length"]],s["features"]) for i,s in enumerate(samples))
        interfaces.append(dict(split=split,real_lengths=b["real_lengths"],feature_lengths=b["lengths"].tolist(),batch_shape=list(b["features"].shape),
                               target_shape=list(b["target"].shape),Mask_correct=True,model_forward_calls=0))
    write_json(OUT/"reader_interface_checks.json",dict(status="passed",splits=interfaces,parameters_updated=False))
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
