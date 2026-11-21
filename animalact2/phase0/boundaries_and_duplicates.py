"""Diagnostic counts and duplicate candidates only; no sample construction."""
from collections import Counter,defaultdict
import csv,json,math,itertools
from pathlib import Path
import numpy as np
from audit_dataset import ROOT,OUT,DATA,guard,read,save,csvsave,sha,TRACKED
import sys


def main():
    sys.addaudithook(guard)
    manifest=read(DATA/"manifest.json");rows=manifest["samples"];names=read(DATA/"action_classes.json")["classes"]
    lookup={c["name"]:c["class_id"] for c in names};counts=Counter();atomic=Counter();byclass={s:Counter() for s in ("train","valid","test")};examples=[];frame_anomalies=[]
    for row in rows:
        ann=read(DATA/row["annotation_path"]);n=row["nb_frames"];fps=row["fps"];spans=[]
        for ai,animal in enumerate(ann["annotations"]):
            for ii,a in enumerate(animal["actions"]):
                start,end=float(a["start"]),float(a["end"]);i,j=a.get("start_frame"),a.get("end_frame")
                provided=isinstance(i,int) and isinstance(j,int) and 0<=i<j<=n and abs(i-start*fps)<=1.05 and abs(j-end*fps)<=1.05
                if provided:lo,hi=i,j;counts["provided_positive_frame_bounds"]+=1
                else:
                    lo=max(0,min(n,math.floor(start*fps+1e-9)));hi=max(0,min(n,math.ceil(end*fps-1e-9)))
                    counts["fallback_exposure_overlap_bounds"]+=1
                    if i is not None and j is not None:
                        counts["provided_nonpositive_or_inconsistent_frame_bounds"]+=1
                        frame_anomalies.append(dict(split=row["split"],video_id=row["video_id"],animal_entry=ai,interval_index=ii,action=a["action"],kind="provided_frame_span_empty_or_inconsistent",severity="review",start=start,end=end,detail=f"source frames[{i},{j}); seconds positive; do not silently drop or rewrite; proposed fallback[{lo},{hi})"))
                point_lo=max(0,min(n,math.ceil(start*fps-1e-9)));point_hi=max(0,min(n,math.ceil(end*fps-1e-9)))
                counts["strict_point_no_frame"]+=point_hi<=point_lo
                counts["proposed_conservative_no_frame"]+=hi<=lo
                if point_hi<=point_lo or (i is not None and j is not None and j<=i):
                    examples.append({"video_id":row["video_id"],"split":row["split"],"action":a["action"],"start":start,"end":end,"source_start_frame":i,"source_end_frame":j,"point_bounds":[point_lo,point_hi],"proposed_frame_bounds":[lo,hi]})
                if hi>lo:spans.append((lo,hi,lookup[a["action"]]))
        bounds=sorted({v for l,h,c in spans for v in (l,h)});atoms=[]
        for lo,hi in zip(bounds,bounds[1:]):
            active=frozenset(c for a,b,c in spans if a<=lo<b)
            if not active:continue
            if atoms and atoms[-1][1]==lo and atoms[-1][2]==active:atoms[-1]=(atoms[-1][0],hi,active)
            else:atoms.append((lo,hi,active))
        for lo,hi,labels in atoms:
            s=row["split"];atomic[s+"_atoms"]+=1;atomic[s+"_multilabel"]+=len(labels)>1;atomic[s+"_under4frames"]+=hi-lo<4;atomic[s+"_under16frames"]+=hi-lo<16;byclass[s].update(labels)
    with (OUT/"annotation_anomalies.csv").open(encoding="utf-8-sig") as f:old_anomalies=[r for r in csv.DictReader(f) if r["kind"]!="provided_frame_span_empty_or_inconsistent"]
    csvsave("annotation_anomalies.csv",old_anomalies+frame_anomalies)
    main_ids=read(DATA/"main_evaluation_classes.json")["class_ids"]
    save("evidence/boundary_policy_analysis.json",{"counts":dict(counts),"examples":examples,"proposed_frame_grid_counts_only":dict(atomic),"supported_classes_proposed":{s:len(c) for s,c in byclass.items()},"main31_supported_proposed":{s:sum(i in c for i in main_ids) for s,c in byclass.items()},"policy":"Design candidate: use provided positive consistent half-open frame bounds; otherwise derive frame exposure overlap [floor(start*fps),ceil(end*fps)), clamp to video. Compare against seconds-first point timestamps. No source field is changed, no clip manifest created."})
    summary=read(OUT/"evidence/annotation_summary.json")
    summary["anomaly_kinds"]=dict(Counter(r["kind"] for r in old_anomalies+frame_anomalies))
    summary["anomaly_severity"]=dict(Counter(r["severity"] for r in old_anomalies+frame_anomalies))
    save("evidence/annotation_summary.json",summary)
    fingerprints=read(OUT/"evidence/perceptual_frame_fingerprints.json");ids=sorted(fingerprints);bucket=defaultdict(list);candidates=defaultdict(int)
    for i,vid in enumerate(ids):
        for pos,fp in enumerate(fingerprints[vid]):
            if fp is None or fp["std"]<8 or fp["gradient"]<3:continue
            h=fp["hash"]
            for band in range(8):
                key=(pos,band,(h>>(band*8))&255)
                for j in bucket[key]:candidates[(j,i)]|=1<<pos
                bucket[key].append(i)
    metadata={r["video_id"]:r for r in rows};matches=[]
    for (a,b),mask in candidates.items():
        if mask.bit_count()<3:continue
        left,right=ids[a],ids[b];distances=[]
        for f,g in zip(fingerprints[left],fingerprints[right]):
            distances.append((f["hash"]^g["hash"]).bit_count() if f is not None and g is not None and min(f["std"],g["std"])>=8 and min(f["gradient"],g["gradient"])>=3 else 65)
        hits=sum(d<=4 for d in distances)
        if hits>=3:
            x,y=metadata[left],metadata[right]
            matches.append({"left":left,"right":right,"left_split":x["split"],"right_split":y["split"],"cross_split":x["split"]!=y["split"],"same_published_group":x["split_group_id"]==y["split_group_id"],"same_exact_file_hash":x["video_sha256"]==y["video_sha256"],"matched_frame_positions":hits,"hamming_distances":";".join(map(str,distances)),"status":"perceptual_similarity_candidate_not_recording_identity"})
    matches.sort(key=lambda r:(not r["cross_split"],-r["matched_frame_positions"],r["left"],r["right"]))
    csvsave("evidence/near_duplicate_candidates.csv",matches,["left","right","left_split","right_split","cross_split","same_published_group","same_exact_file_hash","matched_frame_positions","hamming_distances","status"])
    save("evidence/near_duplicate_summary.json",{"videos":len(ids),"method":"5 relative-time frames; 64-bit dHash; >=3 aligned positions with Hamming<=4; low-information frames excluded from candidate generation; LSH8x8 bands","candidate_pairs":len(matches),"cross_split_candidate_pairs":sum(r["cross_split"] for r in matches),"same_published_group_pairs":sum(r["same_published_group"] for r in matches),"limitations":"Not exhaustive; may miss shifted/edited/cropped recordings and may yield similar-scene false positives. No split change and no confirmed source-identity claims."})
    save("evidence/secondary_source_provenance.json",TRACKED)
    print("Frame-boundary diagnostics",dict(counts),dict(atomic),flush=True)
    print("Near-duplicate candidates",len(matches),"cross-split",sum(r["cross_split"] for r in matches),flush=True)


if __name__=="__main__":main()
