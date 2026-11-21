"""All cached arrays, keys, frame order and labels independently checked; no CNN inference."""
from __future__ import annotations
import csv
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
import math
import time
import numpy as np
import torch
from .extract import load_plan
from .storage import CACHE,FrameStore,context,frame_key,clip_descriptor,read_array
from .support import ROOT,OUT,PILOT,enable_guard,sha,digest,write_json,RAMMonitor


def rat(value):return Fraction(str(value))


def frame_points(s,e,n,fps):return set(range(max(0,math.ceil(s*fps)),min(n,math.ceil(e*fps))))


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def validate():
    begin=time.perf_counter();plan,records=load_plan();ctx=context();store=FrameStore(CACHE,ctx)
    expected={frame_key(r,k,ctx) for r in records for k in r["frame_indices"]}
    assert set(store.locations)==expected and len(expected)==73884
    for directory in (CACHE/"frames").glob("shard_*"):
        receipt=json.loads((directory/"receipt.json").read_text(encoding="utf-8"))
        assert receipt["array_dtype"]=="float32" and receipt["compression"]=="none"
        assert 1<=receipt["array_shape"][0]<=8 and receipt["array_shape"][1]==512
        assert len(receipt["frames"])==receipt["array_shape"][0]
    assert len({r["clip_id"] for r in records})==5646
    names={c["name"]:c["class_id"] for c in json.loads((ROOT/"animalact2/release_v1/class_mapping.json").read_text(encoding="utf-8"))["classes"]}
    main_ids=json.loads((ROOT/"animalact2/release_v1/main_evaluation_classes.json").read_text(encoding="utf-8"))["class_ids"]
    groups=defaultdict(list)
    for r in records:groups[(r["split"],r["parent_video_id"])].append(r)
    clip_rows=[];source_rows=[];split_keys=defaultdict(set);classes={s:Counter() for s in ("train","valid")};class_parents={s:defaultdict(set) for s in ("train","valid")}
    failed=[];source_total=0;positive_pairs=0;link_count=0
    with RAMMonitor() as ram:
        for parent_number,((split,vid),rr) in enumerate(sorted(groups.items()),1):
            r0=rr[0];fps=rat(r0["fps"]);n=r0["parent_num_frames"];duration=Fraction(n,1)/fps
            path=ROOT/r0["annotation_path"]
            assert sha(path)==r0["annotation_sha256"]
            raw=json.loads(path.read_text(encoding="utf-8"));assert raw["video_id"]==vid
            annotations=[]
            for ai,animal in enumerate(raw["annotations"]):
                for ii,a in enumerate(animal["actions"]):
                    pointer=f"/annotations/{ai}/actions/{ii}"
                    uid="ann_"+hashlib.sha256(json.dumps([r0["annotation_sha256"],pointer],sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()[:20]
                    s,e=rat(a["start"]),rat(a["end"])
                    effective=(s,min(e,duration)) if s<e and s<duration else (s,e)
                    annotations.append(dict(uid=uid,cid=names[a["action"]],s=effective[0],e=effective[1],os=s,oe=e,
                                            ref=r0["annotation_path"]+"#"+pointer,animal=animal["animal"]))
            by_uid={a["uid"]:a for a in annotations};source_total+=len(annotations)
            for r in rr:
                try:
                    s,e=rat(r["effective_start_exact"]),rat(r["effective_end_exact"])
                    available=sorted(frame_points(s,e,n,fps))
                    selected=available if len(available)<=16 else [available[round(Fraction(j*(len(available)-1),15))] for j in range(16)]
                    assert selected==r["frame_indices"] and len(selected)==r["effective_length"] and r["valid_frame_mask"]==[True]*len(selected)
                    supports=defaultdict(set);contributors=defaultdict(set);overlaps={}
                    for a in annotations:
                        lo,hi=max(s,a["s"]),min(e,a["e"])
                        if lo<hi:
                            supports[a["cid"]].update(frame_points(lo,hi,n,fps));contributors[a["cid"]].add(a["uid"]);overlaps[a["uid"]]=(lo,hi)
                    ids=sorted(contributors)
                    assert ids==r["class_ids"] and r["target_119"]==[int(i in ids) for i in range(119)]
                    assert set(overlaps)==set(r["all_positive_label_source_uids"])
                    for ev in r["label_evidence"]:
                        i=ev["class_id"];assert supports[i] and len(supports[i])==ev["dense_support_count"]
                        hits=[k for k in selected if k in supports[i]]
                        assert ev["sampled_support_indices"]==hits and len(hits)==ev["sampled_support_count"]
                        assert {a["annotation_uid"] for a in ev["contributors"]}==contributors[i]
                        for a in ev["contributors"]:
                            uid=a["annotation_uid"]
                            assert a["annotation_ref"]==by_uid[uid]["ref"] and a["animal_name"]==by_uid[uid]["animal"]
                            assert (rat(a["overlap_start_exact"]),rat(a["overlap_end_exact"]))==overlaps[uid]
                            link_count+=1
                        positive_pairs+=1;classes[split][i]+=1;class_parents[split][i].add(vid)
                    for w in r["original_windows"]:
                        os_,oe=rat(w["original_start_exact"]),rat(w["original_end_exact"])
                        assert os_<oe and os_<duration and (os_,min(oe,duration))==(s,e)
                        assert (rat(w["effective_start_exact"]),rat(w["effective_end_exact"]))==(s,e)
                        for uid in w["source_annotation_uids"]:assert (by_uid[uid]["os"],by_uid[uid]["oe"])==(os_,oe)
                    desc=clip_descriptor(r,ctx);directory=CACHE/"clips"/split/r["clip_id"]
                    array,receipt=read_array(directory,dict(clip_descriptor=desc,clip_key=digest(desc),kind="clip_features",array_dtype="float32",compression="none"))
                    assert array.shape==(len(selected),512) and array.dtype==np.float32 and np.isfinite(array).all()
                    expected_rows=np.stack([store.get(key) for key in desc["frame_keys"]])
                    assert np.array_equal(array,expected_rows),"Clip/unique-frame rows differ"
                    split_keys[split].update(desc["frame_keys"])
                    clip_rows.append(dict(split=split,clip_id=r["clip_id"],parent_video_id=vid,status="passed",effective_length=len(selected),
                                          array_shape=json.dumps(list(array.shape)),frame_indices=json.dumps(selected),class_ids=json.dumps(ids),
                                          target_SHA256=digest(r["target_119"]),label_metadata_SHA256=desc["label_metadata_SHA256"],
                                          array_SHA256=receipt["array_SHA256"],annotation_SHA256=r["annotation_sha256"],
                                          feature_path=(directory/"features.npy").relative_to(OUT).as_posix(),frame_rows_equal=True,source_links_verified=True))
                except Exception as exc:failed.append(dict(split=split,clip_id=r["clip_id"],parent_video_id=vid,error=repr(exc)))
            source_rows.append(dict(split=split,parent_video_id=vid,annotation_records=len(annotations),annotation_SHA256=r0["annotation_sha256"],source_video_SHA256=r0["video_sha256"],clips=len(rr)))
            if parent_number%250==0 or parent_number==len(groups):print(json.dumps(dict(phase="independent_validation",parents=parent_number,clips_verified=len(clip_rows),failures=len(failed),seconds=round(time.perf_counter()-begin,2))),flush=True)
        # Reject extra or misplaced cached Clip directories.
        stored={p.name for s in ("train","valid") for p in (CACHE/"clips"/s).iterdir() if p.is_dir() and not p.name.startswith(".pending_")}
        assert stored=={r["clip_id"] for r in records}
        # Same full cache uses exactly the Pilot keys and FP32 values on all120 Pilot clips.
        pilot_plan=json.loads((PILOT/"selection.json").read_text(encoding="utf-8"));pilot_compared=0
        record_by_id={r["clip_id"]:r for r in records}
        for split,ids in pilot_plan["selections"].items():
            for cid in ids:
                r=record_by_id[cid];desc=clip_descriptor(r,ctx)
                full,_=read_array(CACHE/"clips"/split/cid,dict(clip_descriptor=desc))
                old,_=read_array(PILOT/"cache/clips"/split/cid,dict(clip_descriptor=desc))
                assert np.array_equal(full,old);pilot_compared+=1
    assert pilot_compared==120
    split_rows=[]
    for split,expected_n in (("train",4667),("valid",979)):
        rr=[r for r in records if r["split"]==split];cc=[r for r in clip_rows if r["split"]==split]
        split_rows.append(dict(split=split,expected_clips=expected_n,verified_clips=len(cc),failed_clips=sum(f["split"]==split for f in failed),
                               parent_videos=len({r["parent_video_id"] for r in rr}),unique_frames=len(split_keys[split]),clip_frame_references=sum(r["effective_length"] for r in rr),
                               positive_classes=len(classes[split]),main31_supported=sum(classes[split][i]>0 for i in main_ids),
                               length1_3_clips=sum(r["effective_length"]<=3 for r in rr),length4_15_clips=sum(4<=r["effective_length"]<=15 for r in rr)))
    assert not split_keys["train"]&split_keys["valid"]
    class_rows=[]
    for name,cid in names.items():
        row=dict(class_id=cid,action=name,main31=cid in main_ids)
        for split in ("train","valid"):row.update({split+"_positive_clips":classes[split][cid],split+"_positive_parents":len(class_parents[split][cid])})
        class_rows.append(row)
    assert classes["train"][23]==0 and all(classes[s][i]>0 for s in ("train","valid") for i in main_ids)
    csv_write(OUT/"validation/clip_checks.csv",clip_rows);csv_write(OUT/"validation/source_checks.csv",source_rows)
    csv_write(OUT/"validation/split_statistics.csv",split_rows);csv_write(OUT/"validation/class_support_119.csv",class_rows)
    csv_write(OUT/"validation/main31_support.csv",[r for r in class_rows if r["main31"]])
    write_json(OUT/"validation/errors.json",failed)
    result=dict(status="passed" if not failed and len(clip_rows)==5646 else "failed",split_statistics=split_rows,verified_unique_frames=len(store.locations),verified_clips=len(clip_rows),
                failures=len(failed),all_cached_rows_equal_unique_frame_bank=True,raw_annotation_records_checked=source_total,positive_clip_class_pairs_checked=positive_pairs,
                original_annotation_contributor_links_checked=link_count,Pilot_clip_arrays_bitwise_equal=120,context_identical_to_Pilot=True,
                FP32_uncompressed=True,Defensive_Rearing_Train_positive_clips=0,main31_unchanged=True,
                elapsed_seconds=time.perf_counter()-begin,RAM=ram.result(),Test_read=False,CNN_inference_in_validation=0,training_started=False)
    write_json(OUT/"validation_summary.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if result["status"]!="passed":raise CacheIntegrityError("Full validation failed; inspect retained errors")
    return result


if __name__=="__main__":enable_guard();validate()
