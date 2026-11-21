"""Train/Valid cache-only reads: no source-video opens or encoder construction."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from animalact2.phase1_5.data import EpochShuffleSampler
from animalact2.phase2_2.data import collate_features as pilot_collate
from .support import FULL,sha,digest,worker_init


class CachedFeatureDataset(Dataset):
    def __init__(self,split,*,clip_ids=None):
        if split not in ("train","valid"):raise ValueError("Train/Valid only")
        assert json.loads((FULL/"validation_summary.json").read_text())["status"]=="passed"
        self.source=ReleaseClipDataset(split,clip_ids=clip_ids);self.records=self.source.records;self.split=split
        self.ctx=json.loads((FULL/"cache/context.json").read_text(encoding="utf-8"));self.context_hash=digest(self.ctx)
        plan=json.loads((FULL/"full_plan.json").read_text(encoding="utf-8"))
        assert self.ctx==plan["context"] and self.source.manifest_sha256==plan["manifest_SHA256"][split]
        assert set(r["clip_id"] for r in self.records)<=set(plan["selections"][split])

    def __len__(self):return len(self.records)

    def __getitem__(self,i):
        r=self.records[i];d=FULL/"cache"/self.split/r["clip_id"]
        receipt=json.loads((d/"receipt.json").read_text(encoding="utf-8"))
        expected=dict(context_SHA256=self.context_hash,clip_id=r["clip_id"],split=self.split,parent_video_id=r["parent_video_id"],
                      video_path=r["video_path"],video_content_SHA256=r["video_sha256"],frame_indices=r["frame_indices"],real_length=r["effective_length"],
                      effective_start=r["effective_start_exact"],effective_end=r["effective_end_exact"],
                      label_and_provenance_SHA256=digest(dict(target=r["target_119"],evidence=r["label_evidence"])))
        assert receipt["descriptor"]==expected and receipt["key"]==digest(expected)
        assert sha(d/"features.npy")==receipt["features_SHA256"]
        x=np.load(d/"features.npy",allow_pickle=False)
        assert x.dtype==np.float32 and x.shape==((r["effective_length"]+1)//2,768) and np.isfinite(x).all()
        assert receipt["target_119"]==r["target_119"] and receipt["label_evidence"]==r["label_evidence"]
        assert [v for pair in receipt["tubelet_frame_indices"] for v in pair if v is not None]==r["frame_indices"]
        assert receipt["valid_tubelet_mask"]==[True]*len(x)
        return dict(features=torch.from_numpy(x.copy()),length=len(x),real_length=r["effective_length"],target=torch.tensor(r["target_119"],dtype=torch.float32),
                    clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],split=self.split,frame_indices=r["frame_indices"],metadata=r)


def collate_features(samples):
    result=pilot_collate(samples)
    # Shared trainer has a generic 'frames' input slot; it carries 768-D features here.
    result.update(frames=result.pop("features"),input_kind="frozen_videomae_temporal_features",parent_video_ids=[s["parent_video_id"] for s in samples])
    return result


def make_feature_loader(ds,config,*,training,workers_override=None):
    if ds.split!=("train" if training else "valid"):raise ValueError("Loader split mismatch")
    d=config["data"];workers=d["num_workers"] if workers_override is None else workers_override
    args=dict(dataset=ds,batch_size=d["batch_size"] if training else d["valid_batch_size"],
              sampler=EpochShuffleSampler(ds,config["seed"]) if training else None,shuffle=False,drop_last=False,
              num_workers=workers,pin_memory=d["pin_memory"],collate_fn=collate_features,worker_init_fn=worker_init,
              generator=torch.Generator().manual_seed(config["seed"]+(0 if training else 17)))
    if workers:args.update(prefetch_factor=d["prefetch_factor"],persistent_workers=True)
    return DataLoader(**args)
