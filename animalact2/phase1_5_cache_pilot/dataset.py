"""Shared FP32 features; labels and lineage always come from the sealed Manifest."""
from __future__ import annotations
import copy
import json
import torch
from torch.utils.data import Dataset,DataLoader
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from animalact2.phase1_5.data import EpochShuffleSampler
from .cache import CACHE,context,select_plan,read_array,clip_descriptor
from .support import ROOT,enable_guard,digest


class CachedFeatureDataset(Dataset):
    def __init__(self,split,*,clip_ids=None):
        if split not in ("train","valid"):raise ValueError("Pilot only permits Train/Valid")
        self.plan=select_plan();self.split=split;self.ctx=context()
        selected=self.plan["selections"][split] if clip_ids is None else list(clip_ids)
        if not set(selected)<=set(self.plan["selections"][split]):raise ValueError("Cannot expand the locked 120-clip pilot")
        self.source=ReleaseClipDataset(split,clip_ids=selected)
        if self.source.manifest_sha256!=self.plan["manifest_SHA256"][split]:raise ValueError("Source Manifest changed after pilot selection")
        self.records=self.source.records

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index];saved=self.plan["source_stats"][r["parent_video_id"]];st=(ROOT/r["video_path"]).stat()
        if (st.st_size,st.st_mtime_ns)!=(saved["bytes"],saved["mtime_ns"]):raise ValueError("Source video identity/stat changed")
        desc=clip_descriptor(r,self.ctx)
        array,receipt=read_array(CACHE/"clips"/self.split/r["clip_id"],dict(clip_descriptor=desc,clip_key=digest(desc),kind="clip_features"))
        if tuple(array.shape)!=(r["effective_length"],512):raise ValueError("Unexpected variable-length clip feature shape")
        return dict(features=torch.from_numpy(array.copy()),target=torch.tensor(r["target_119"],dtype=torch.float32),
                    frame_indices=torch.tensor(r["frame_indices"],dtype=torch.int64),effective_length=r["effective_length"],
                    clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],split=self.split,
                    metadata=copy.deepcopy(r),label_sources=copy.deepcopy(r["label_evidence"]))


def collate_features(samples):
    if not samples:raise ValueError("Empty feature batch")
    lengths=torch.tensor([s["effective_length"] for s in samples],dtype=torch.int64)
    if torch.any(lengths<1) or torch.any(lengths>16):raise ValueError("Wrong actual clip length")
    t=int(lengths.max());features=torch.zeros(len(samples),t,512,dtype=torch.float32)
    valid=torch.arange(t)[None,:]<lengths[:,None];indices=torch.full((len(samples),t),-1,dtype=torch.int64)
    for i,s in enumerate(samples):
        n=int(lengths[i])
        if s["features"].dtype!=torch.float32 or s["features"].shape!=(n,512) or not torch.isfinite(s["features"]).all():raise ValueError("Invalid FP32 feature sequence")
        if s["target"].shape!=(119,) or s["frame_indices"].shape!=(n,):raise ValueError("Label/frame mapping mismatch")
        features[i,:n]=s["features"];indices[i,:n]=s["frame_indices"]
    # 'frames' is the existing trainer's generic input slot; input_kind disambiguates.
    return dict(frames=features,input_kind="frozen_resnet18_features",lengths=lengths,valid_frame_mask=valid,padding_mask=~valid,
                frame_indices=indices,target=torch.stack([s["target"] for s in samples]),clip_ids=[s["clip_id"] for s in samples],
                parent_video_ids=[s["parent_video_id"] for s in samples],splits=[s["split"] for s in samples],
                metadata=[s["metadata"] for s in samples],label_sources=[s["label_sources"] for s in samples])


def seed_worker(worker_id):
    enable_guard();torch.set_num_threads(1)


def make_pilot_loader(dataset,config,*,training,cached,workers=None):
    from animalact2.phase1_5_1.collate import collate_clips
    if dataset.split!=("train" if training else "valid"):raise ValueError("Train/Valid loader split mismatch")
    n=config["data"]["num_workers"] if workers is None else workers
    sampler=EpochShuffleSampler(dataset,config["seed"]) if training else None
    if sampler:sampler.set_epoch(1)
    args=dict(dataset=dataset,batch_size=config["data"]["batch_size"] if training else config["data"]["valid_batch_size"],
              sampler=sampler,shuffle=False,num_workers=n,drop_last=False,pin_memory=config["data"]["pin_memory"],
              collate_fn=collate_features if cached else collate_clips,worker_init_fn=seed_worker,
              generator=torch.Generator().manual_seed(config["seed"]+(0 if training else 17)))
    if n:args.update(prefetch_factor=config["data"]["prefetch_factor"],persistent_workers=True)
    return DataLoader(**args)


class FeatureForwardView:
    """Call adapter with no parameters/layers; original model/state/architecture unchanged."""
    def __init__(self,model):self.model=model
    def __call__(self,features,lengths,valid_frame_mask):return self.model.forward_features(features,lengths,valid_frame_mask)
    def train(self,mode=True):self.model.train(mode);return self
    def eval(self):self.model.eval();return self
    def parameters(self):return self.model.parameters()
    def named_parameters(self):return self.model.named_parameters()
