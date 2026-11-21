"""Same Pilot context/key/NPY/receipt, with independent full-cache output paths."""
import json
import os
from pathlib import Path
import time
import numpy as np
from animalact2.phase2_2.data import cache_context,descriptor
from .support import OUT,PILOT,sha,digest

CACHE=OUT/"cache"


def read_cache(r,ctx,*,root=CACHE):
    d=Path(root)/r["split"]/r["clip_id"]
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
    g=receipt["geometry"]
    assert g["real_hw"]==[224,398] and g["aligned_hw"]==[224,400] and g["grid"]==[len(x),14,25]
    assert g["real_length"]==r["effective_length"] and g["aligned_length"]==r["effective_length"]+(r["effective_length"]%2)
    return x,receipt


def commit(r,ctx,features,geometry):
    desc=descriptor(r,ctx);d=CACHE/r["split"]/r["clip_id"]
    if d.exists():raise FileExistsError("Never overwrite committed full features")
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


def reuse_pilot(r,ctx):
    previous=read_cache(r,ctx,root=PILOT/"cache")
    if previous is None:return False
    a,receipt=previous
    commit(r,ctx,a,receipt["geometry"])
    current=read_cache(r,ctx)
    assert np.array_equal(a,current[0]) and receipt==current[1]
    assert sha(PILOT/"cache"/r["split"]/r["clip_id"]/"features.npy")==sha(CACHE/r["split"]/r["clip_id"]/"features.npy")
    return True
