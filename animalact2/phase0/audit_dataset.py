"""Phase0: independently inspect the supplied dataset; never construct training data."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/"AnimalAct"
SPLITS=("train","valid","test")
TRACKED={}


def guard(event,args):
    if event!="open" or not isinstance(args[0],(str,bytes)):return
    path=Path(os.fsdecode(args[0])).resolve()
    if not path.is_relative_to(ROOT):return
    rel=path.relative_to(ROOT).as_posix().casefold()
    if not path.is_relative_to(OUT) and ("model6" in rel or "predictions" in rel or path.suffix in (".pt",".safetensors")):
        raise RuntimeError("Phase0 excludes predictions, weights and feature tensors: "+rel)
    mode,flags=args[1],args[2]
    write=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and bool(flags&3))
    if write and not path.is_relative_to(OUT):raise RuntimeError("Historical write blocked: "+rel)


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def read(path):
    path=Path(path);b=path.read_bytes()
    if not path.is_relative_to(OUT):TRACKED[path.relative_to(ROOT).as_posix()]={"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
    return json.loads(b.decode("utf-8-sig"))


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def csvsave(name,rows,fields=None):
    fields=fields or list(rows[0])
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def describe(values):
    a=np.asarray(values,dtype=float)
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"p10":float(np.quantile(a,.1)),"p90":float(np.quantile(a,.9)),"min":float(a.min()),"max":float(a.max())} if len(a) else {"n":0}


def video_inspect(row):
    path=DATA/row["video_path"];before=path.stat();actual_sha=sha(path)
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_NONE])
    result={"video_id":row["video_id"],"split":row["split"],"video_path":"AnimalAct/"+row["video_path"],"bytes":before.st_size,"sha256":actual_sha,"hash_matches_manifest":actual_sha==row["video_sha256"],
            "opened":cap.isOpened(),"decoded_frames":0,"read_status":"not_opened","header_frames":0,"width":0,"height":0,"fps":0.,"fourcc":"","duration_seconds":0.,"frame_shape_changes":0}
    hashes=[]
    try:
        if cap.isOpened():
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));code=int(cap.get(cv2.CAP_PROP_FOURCC))
            result.update(header_frames=n,fps=fps,width=w,height=h,fourcc="".join(chr((code>>(8*i))&255) for i in range(4)),duration_seconds=n/fps if fps>0 else 0)
            wanted=np.rint(np.linspace(0,max(0,n-1),5)).astype(int).tolist();want=set(wanted);small={};index=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame is None or frame.size==0:break
                if frame.shape[:2]!=(h,w):result["frame_shape_changes"]+=1
                if index in want:
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    tiny=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
                    bits=(tiny[:,1:]>tiny[:,:-1]).ravel()
                    value=0
                    for bit in bits:value=(value<<1)|int(bit)
                    small[index]={"hash":value,"std":float(gray.std()),"gradient":float(np.abs(tiny[:,1:].astype(float)-tiny[:,:-1]).mean())}
                index+=1
            result["decoded_frames"]=index
            result["read_status"]="complete_decode_matches_header" if n>0 and index==n else "frame_count_or_decode_failure"
            hashes=[small.get(i) for i in wanted]
    except Exception as exc:result["read_status"]="error: "+str(exc)
    finally:cap.release()
    after=path.stat();result["size_mtime_unchanged"]=before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
    result["header_matches_manifest"]=result["header_frames"]==row["nb_frames"] and abs(result["fps"]-row["fps"])<1e-5
    return result,hashes


def main():
    start=time.perf_counter();sys.addaudithook(guard);cv2.setNumThreads(1)
    manifest=read(DATA/"manifest.json");vocab=read(DATA/"action_classes.json");main=read(DATA/"main_evaluation_classes.json")
    splitmeta=read(DATA/"splits/split_manifest.json");olddup=read(DATA/"splits/duplicate_audit.json")
    rows=manifest["samples"];names=[x["name"] for x in vocab["classes"]];lookup={n:i for i,n in enumerate(names)}
    assert len(rows)==len({r["video_id"] for r in rows})==2968 and len(names)==119
    anomalies=[];stats={i:{s:{"intervals":0,"videos":set(),"animals":set(),"seconds":[],"no_frame":0,"uniform_missed":0} for s in SPLITS} for i in range(119)}
    schema={"root":Counter(),"animal":Counter(),"interval":Counter()};interval_rows=[];pervideo=[];positive_pairs=[]
    totals=Counter();examples={};atomic=Counter();atomic_byclass={s:Counter() for s in SPLITS};no_frame_classes=defaultdict(set)
    for s in SPLITS:
        part=read(DATA/"splits"/(s+".json"));expected=[r for r in rows if r["split"]==s]
        assert {r["video_id"] for r in part["samples"]}=={r["video_id"] for r in expected}
        files={p.stem for p in (DATA/"videos"/s).glob("*.mp4")};anns={p.stem for p in (DATA/"annotations"/s).glob("*.json")}
        assert files==anns=={r["video_id"] for r in expected}
    for row in rows:
        s=row["split"];vid=row["video_id"];ann=read(DATA/row["annotation_path"]);n=row["nb_frames"];fps=row["fps"];duration=n/fps
        assert sha(DATA/row["annotation_path"])==row["annotation_sha256"] and ann["video_id"]==vid and ann["video"]==row["video"]
        schema["root"].update(ann.keys());all_actions=[];animal_action_sets=[];label_set=set();byclass=defaultdict(list)
        for ai,animal in enumerate(ann["annotations"]):
            schema["animal"].update(animal.keys());aset=set()
            for ii,action in enumerate(animal["actions"]):
                schema["interval"].update(action.keys());name=action["action"];cid=lookup[name];a=float(action["start"]);b=float(action["end"])
                first=max(0,min(n,math.ceil(a*fps-1e-9)));stop=max(0,min(n,math.ceil(b*fps-1e-9)))
                valid=math.isfinite(a) and math.isfinite(b) and 0<=a<b and b<=duration+1/fps+1e-4
                if not valid:anomalies.append(dict(split=s,video_id=vid,animal_entry=ai,interval_index=ii,action=name,kind="invalid_seconds",severity="error",start=a,end=b,detail="Check negative/nonpositive/over1frame bounds"))
                if b>duration+1e-6:anomalies.append(dict(split=s,video_id=vid,animal_entry=ai,interval_index=ii,action=name,kind="subframe_end_overrun" if b-duration<=1/fps+1e-4 else "large_end_overrun",severity="info" if b-duration<=1/fps+1e-4 else "error",start=a,end=b,detail=f"end-video_duration={b-duration:.9f}s; original untouched"))
                if stop<=first:
                    anomalies.append(dict(split=s,video_id=vid,animal_entry=ai,interval_index=ii,action=name,kind="no_frame_timestamp_inside_interval",severity="review",start=a,end=b,detail="No original frame k/fps satisfies start<=t<end; no artificial frames added"));no_frame_classes[s].add(name)
                for field,t in [("start_frame",a),("end_frame",b)]:
                    if field not in action:anomalies.append(dict(split=s,video_id=vid,animal_entry=ai,interval_index=ii,action=name,kind="optional_frame_field_missing",severity="info",start=a,end=b,detail=field))
                    elif abs(float(action[field])-t*fps)>1.05:anomalies.append(dict(split=s,video_id=vid,animal_entry=ai,interval_index=ii,action=name,kind="frame_time_disagreement",severity="error",start=a,end=b,detail=field))
                st=stats[cid][s];st["intervals"]+=1;st["videos"].add(vid);st["animals"].add(animal["animal"]);st["seconds"].append(b-a);st["no_frame"]+=stop<=first
                label_set.add(cid);aset.add(cid);byclass[cid].append((a,b));all_actions.append((a,b,cid,ai,animal["animal"]))
                interval_rows.append(dict(split=s,video_id=vid,animal_entry=ai,animal_name=animal["animal"],action_id=cid,action=name,start=a,end=b,start_frame=action.get("start_frame"),end_frame=action.get("end_frame"),frame_support=stop-first,seconds=b-a))
            animal_action_sets.append(aset)
        totals[s+"_videos"]+=1;totals[s+"_intervals"]+=len(all_actions)
        multi=len(label_set)>=2;totals["multi_action_videos"]+=multi;totals["empty_action_videos"]+=not label_set
        overlap=False;diff_animal=False;main025=False
        for i,l in enumerate(all_actions):
            for r in all_actions[i+1:]:
                intersection=min(l[1],r[1],duration)-max(l[0],r[0],0.)
                if intersection>1e-9:
                    if l[2]!=r[2]:overlap=True
                    if l[4]!=r[4]:diff_animal=True
                    if intersection>=.25-1e-9 and l[2]!=r[2] and l[2] in main["class_ids"] and r[2] in main["class_ids"]:main025=True
        totals["distinct_action_overlap_videos"]+=overlap;totals["different_animal_name_overlap_videos"]+=diff_animal
        totals[s+"_main025_concurrent_videos"]+=main025
        for key,condition in [("multiple_intervals",len(all_actions)>1),("multiple_action_labels",multi),("distinct_action_overlap",overlap),("different_animal_name_overlap",diff_animal),("same_animal_entry_multiple_actions",any(len(a)>1 for a in animal_action_sets))]:
            if condition and key not in examples:examples[key]={"video_id":vid,"split":s,"annotation_path":"AnimalAct/"+row["annotation_path"],"annotations":ann["annotations"]}
        uniform=np.rint(np.linspace(0,n-1,16))/fps
        for cid,spans in byclass.items():
            hit=any(np.any((uniform>=a-1e-9)&(uniform<b-1e-9)) for a,b in spans)
            stats[cid][s]["uniform_missed"]+=not hit
            positive_pairs.append({"split":s,"video_id":vid,"action_id":cid,"action":names[cid],"uniform16_hits_interval":hit,"main_evaluation":cid in main["class_ids"]})
        bounds=sorted({max(0.,min(duration,t)) for a,b,*_ in all_actions for t in (a,b)})
        segments=[]
        for a,b in zip(bounds,bounds[1:]):
            midpoint=(a+b)/2;active=frozenset(cid for start,end,cid,*_ in all_actions if start<=midpoint<end)
            if not active or b<=a:continue
            if segments and segments[-1][2]==active and abs(segments[-1][1]-a)<1e-9:segments[-1]=(segments[-1][0],b,active)
            else:segments.append((a,b,active))
        for a,b,active in segments:
            atomic[s+"_nonempty_atoms"]+=1;atomic[s+"_multilabel_atoms"]+=len(active)>1;atomic[s+"_under025_atoms"]+=b-a<.25
            atomic[s+"_under050_atoms"]+=b-a<.5;atomic[s+"_no_frame_atoms"]+=math.ceil(b*fps-1e-9)<=math.ceil(a*fps-1e-9)
            atomic_byclass[s].update(active)
        pervideo.append({"video_id":vid,"split":s,"video_path":"AnimalAct/"+row["video_path"],"annotation_path":"AnimalAct/"+row["annotation_path"],"intervals":len(all_actions),"action_labels":len(label_set),"animal_name_entries":len(ann["annotations"]),"distinct_animal_names":len({a["animal"] for a in ann["annotations"]}),"different_action_overlap":overlap,"different_animal_overlap":diff_animal,"main025_concurrent":main025,"split_group_id":row["split_group_id"],"planned_atom_count_only":len(segments)})
    assert sum(totals[s+"_videos"] for s in SPLITS)==2968
    statistics=[]
    for cid,name in enumerate(names):
        r={"class_id":cid,"action":name,"main_evaluation":cid in main["class_ids"],"total_instances":sum(stats[cid][s]["intervals"] for s in SPLITS),"total_videos":sum(len(stats[cid][s]["videos"]) for s in SPLITS),"animal_names":"; ".join(sorted(set.union(*(stats[cid][s]["animals"] for s in SPLITS))))}
        for s in SPLITS:
            st=stats[cid][s];r.update({s+"_instances":st["intervals"],s+"_videos":len(st["videos"]),s+"_animal_names":len(st["animals"]),s+"_no_frame_instances":st["no_frame"],s+"_uniform16_missed_video_class_pairs":st["uniform_missed"],s+"_duration_seconds_median":float(np.median(st["seconds"])) if st["seconds"] else None,s+"_planned_atoms":atomic_byclass[s][cid]})
        assert r["total_instances"]==vocab["classes"][cid]["interval_count"] and r["total_videos"]==vocab["classes"][cid]["video_count"]
        statistics.append(r)
    csvsave("annotation_statistics.csv",statistics)
    csvsave("annotation_anomalies.csv",anomalies,["split","video_id","animal_entry","interval_index","action","kind","severity","start","end","detail"])
    csvsave("evidence/action_intervals.csv",interval_rows)
    csvsave("evidence/uniform16_label_coverage.csv",positive_pairs)
    csvsave("evidence/video_annotation_index.csv",pervideo)
    save("evidence/annotation_summary.json",{"totals":dict(totals),"schema_field_occurrences":{k:dict(v) for k,v in schema.items()},"anomaly_kinds":dict(Counter(a["kind"] for a in anomalies)),"anomaly_severity":dict(Counter(a["severity"] for a in anomalies)),"example_files":examples,"planned_atomic_segment_counts_only":dict(atomic),"classes_without_positive_intervals":[names[i] for i in range(119) if not sum(stats[i][s]["intervals"] for s in SPLITS)],"classes_with_no_frame_support_intervals_by_split":{s:sorted(v) for s,v in no_frame_classes.items()},"dataset_revision":manifest["dataset_revision"]})
    print("Annotation audit complete; starting CPU full decoding and SHA verification of2968 videos.",flush=True)
    results=[];fingerprints={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(video_inspect,row):row for row in rows}
        for i,job in enumerate(as_completed(jobs),1):
            result,fp=job.result();results.append(result);fingerprints[result["video_id"]]=fp
            if i%100==0 or i==len(rows):print(f"CPU full video decode/hash {i}/{len(rows)}; elapsed {time.perf_counter()-start:.1f}s",flush=True)
    results.sort(key=lambda r:(r["split"],r["video_id"]))
    csvsave("evidence/video_integrity.csv",results)
    save("evidence/perceptual_frame_fingerprints.json",fingerprints)
    exact=defaultdict(list)
    for r in results:exact[r["sha256"]].append({"video_id":r["video_id"],"split":r["split"]})
    save("evidence/exact_duplicate_groups.json",[v for v in exact.values() if len(v)>1])
    splitstats=[]
    for s in SPLITS:
        selected=[r for r in rows if r["split"]==s];videos=[r for r in results if r["split"]==s]
        splitstats.append({"split":s,"videos":len(videos),"intervals":totals[s+"_intervals"],"unique_detected_groups":len({r["split_group_id"] for r in selected}),"classes_supported":sum(stats[i][s]["intervals"]>0 for i in range(119)),"main31_supported":sum(stats[i][s]["intervals"]>0 for i in main["class_ids"]),"total_video_bytes":sum(v["bytes"] for v in videos),"duration_seconds_mean":float(np.mean([v["duration_seconds"] for v in videos])),"duration_seconds_median":float(np.median([v["duration_seconds"] for v in videos])),"complete_decodes":sum(v["read_status"]=="complete_decode_matches_header" for v in videos),"hash_matches":sum(v["hash_matches_manifest"] for v in videos),"main025_concurrent_videos":totals[s+"_main025_concurrent_videos"]})
    csvsave("split_statistics.csv",splitstats)
    groupmap=defaultdict(set)
    for row in rows:groupmap[row["split_group_id"]].add(row["split"])
    summary={"videos":len(results),"decoded_frames":sum(v["decoded_frames"] for v in results),"total_bytes":sum(v["bytes"] for v in results),"decode_failures":[v for v in results if v["read_status"]!="complete_decode_matches_header"],"hash_mismatches":[v for v in results if not v["hash_matches_manifest"]],"metadata_mismatches":[v for v in results if not v["header_matches_manifest"]],"resolutions":dict(Counter(f"{v['width']}x{v['height']}" for v in results)),"fourcc":dict(Counter(v["fourcc"] for v in results)),"fps":dict(Counter(str(v["fps"]) for v in results)),"duration_seconds":describe([v["duration_seconds"] for v in results]),"cross_split_exact_duplicate_groups":[v for v in exact.values() if len({x["split"] for x in v})>1],"cross_split_published_groups":[k for k,v in groupmap.items() if len(v)>1],"published_split_group_count":len(groupmap),"source_grouping":splitmeta["grouping"],"published_duplicate_links":len(olddup["links"]),"published_fingerprint_records":len(olddup["frame_fingerprints"]),"seconds":time.perf_counter()-start}
    save("evidence/dataset_integrity_summary.json",summary)
    changed=[key for key,value in TRACKED.items() if sha(ROOT/key)!=value["sha256"]]
    assert not changed and all(v["size_mtime_unchanged"] for v in results)
    save("evidence/source_provenance.json",TRACKED)
    save("evidence/audit_execution.json",{"complete":True,"tracked_json_sources":len(TRACKED),"source_hash_changes":changed,"video_full_hash_check":True,"video_decode":"all frames, CPU4workers, no video rewrite","model_calls":0,"training":False,"feature_cache_generation":False,"test_predictions_read":False,"output_scope":str(OUT),"seconds":time.perf_counter()-start})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
