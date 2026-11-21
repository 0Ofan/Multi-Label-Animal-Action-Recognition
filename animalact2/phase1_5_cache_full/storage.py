"""Pilot schema/key/reader reused verbatim; commit target is the new full-cache root."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid
import numpy as np
from animalact2.phase1_5_cache_pilot.cache import (
    context,frame_descriptor,frame_key,label_metadata,clip_descriptor,read_array,CacheIntegrityError,
    FrameStore as PilotFrameStore,
)
from .support import OUT,write_json,sha,digest

CACHE=OUT/"cache"


def commit_array(directory,array,receipt):
    """Same Pilot atomic-directory/NPY/receipt protocol, with a distinct output boundary."""
    directory=Path(directory).resolve()
    if not directory.is_relative_to(OUT):raise ValueError("Full-cache-only output")
    if directory.exists():raise FileExistsError("Never overwrite committed features")
    if array.dtype!=np.float32 or not np.isfinite(array).all():raise CacheIntegrityError("Finite FP32 required")
    staging=directory.parent/(".pending_"+uuid.uuid4().hex);staging.mkdir(parents=True,exist_ok=False)
    with (staging/"features.npy").open("wb") as f:
        np.save(f,np.ascontiguousarray(array),allow_pickle=False);f.flush();os.fsync(f.fileno())
    receipt=dict(receipt,array_SHA256=sha(staging/"features.npy"),array_bytes=(staging/"features.npy").stat().st_size,
                 array_shape=list(array.shape),array_dtype="float32",compression="none")
    write_json(staging/"receipt.json",receipt)
    staging.rename(directory)


class FrameStore(PilotFrameStore):
    def __init__(self,root,ctx):
        root=Path(root)
        if not (root/"context.json").exists():write_json(root/"context.json",ctx)
        # Identical Pilot discovery, SHA verification, duplicate rejection and lookup.
        super().__init__(root,ctx)

    def commit(self,descriptors,array):
        keys=[digest(d) for d in descriptors]
        if len(set(keys))!=len(keys) or any(k in self.locations for k in keys):raise CacheIntegrityError("Duplicate unique-frame commit")
        directory=self.root/"frames"/("shard_"+digest(keys))
        commit_array(directory,array,dict(context_SHA256=self.context_hash,kind="unique_frame_shard",frames=descriptors))
        for row,key in enumerate(keys):self.locations[key]=(directory,row)
