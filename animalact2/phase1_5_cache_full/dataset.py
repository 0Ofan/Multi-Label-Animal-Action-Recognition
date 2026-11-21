"""The same variable-length cached input interface, now for complete Train/Valid."""
from __future__ import annotations
import copy
import json
import torch
from torch.utils.data import Dataset,DataLoader
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from animalact2.phase1_5.data import EpochShuffleSampler
from animalact2.phase1_5_cache_pilot.dataset import collate_features,FeatureForwardView
from .storage import CACHE,context,clip_descriptor,read_array,CacheIntegrityError
from .support import ROOT,OUT,digest,enable_guard


class CachedFeatureDataset(Dataset):
    def __init__(self,split,*,clip_ids=None,require_validated_cache=True):
        if split not in ("train","valid"):raise ValueError("Only fixed Train/Valid are available; Test is not enabled")
        if require_validated_cache:
            status=json.loads((OUT/"validation_summary.json").read_text(encoding="utf-8"))
            if status["status"]!="passed":raise CacheIntegrityError("Cache has not passed full validation")
        self.plan=json.loads((OUT/"full_plan.json").read_text(encoding="utf-8"));self.ctx=context();self.split=split
        if self.plan["context"]!=self.ctx:raise CacheIntegrityError("Current context differs from cached preprocessing/backbone")
        selected=self.plan["selections"][split] if clip_ids is None else list(clip_ids)
        if len(selected)!=len(set(selected)) or not set(selected)<=set(self.plan["selections"][split]):raise ValueError("Unknown or duplicate Clip ID")
        self.source=ReleaseClipDataset(split,clip_ids=selected);self.records=self.source.records
        if self.source.manifest_sha256!=self.plan["manifest_SHA256"][split]:raise CacheIntegrityError("Source Manifest changed")

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index];saved=self.plan["source_stats"][r["parent_video_id"]];st=(ROOT/r["video_path"]).stat()
        if (st.st_size,st.st_mtime_ns)!=(saved["bytes"],saved["mtime_ns"]):raise CacheIntegrityError("Source video identity/stat changed")
        desc=clip_descriptor(r,self.ctx)
        array,_=read_array(CACHE/"clips"/self.split/r["clip_id"],dict(clip_descriptor=desc,clip_key=digest(desc),kind="clip_features"))
        if tuple(array.shape)!=(r["effective_length"],512):raise CacheIntegrityError("Wrong actual feature sequence length")
        return dict(features=torch.from_numpy(array.copy()),target=torch.tensor(r["target_119"],dtype=torch.float32),
                    frame_indices=torch.tensor(r["frame_indices"],dtype=torch.int64),effective_length=r["effective_length"],
                    clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],split=self.split,
                    metadata=copy.deepcopy(r),label_sources=copy.deepcopy(r["label_evidence"]))


def seed_worker(worker_id):enable_guard();torch.set_num_threads(1)


def make_feature_loader(dataset,config,*,training):
    if dataset.split!=("train" if training else "valid"):raise ValueError("Train/Valid loader role mismatch")
    d=config["data"];sampler=EpochShuffleSampler(dataset,config["seed"]) if training else None
    args=dict(dataset=dataset,batch_size=d["batch_size"] if training else d["valid_batch_size"],sampler=sampler,shuffle=False,
              drop_last=False,num_workers=d["num_workers"],pin_memory=d["pin_memory"],collate_fn=collate_features,
              worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(config["seed"]+(0 if training else 17)))
    if d["num_workers"]:args.update(prefetch_factor=d["prefetch_factor"],persistent_workers=True)
    return DataLoader(**args)


__all__=["CachedFeatureDataset","collate_features","FeatureForwardView","make_feature_loader"]
