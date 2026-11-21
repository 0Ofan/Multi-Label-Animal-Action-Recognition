"""Development-only full-cache interface, with the unchanged Pilot collate contract."""
import json
import torch
from torch.utils.data import Dataset
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from animalact2.phase2_2.data import collate_features
from .support import OUT
from .storage import cache_context,read_cache


class CachedFeatureDataset(Dataset):
    def __init__(self,split,*,require_validated=True):
        if split not in ("train","valid"):raise ValueError("Only fixed Train/Valid are available")
        if require_validated:
            assert json.loads((OUT/"validation_summary.json").read_text())["status"]=="passed"
        self.source=ReleaseClipDataset(split);self.records=self.source.records;self.split=split;self.context=cache_context()
        plan=json.loads((OUT/"full_plan.json").read_text(encoding="utf-8"))
        assert self.context==plan["context"] and self.source.manifest_sha256==plan["manifest_SHA256"][split]
        assert [r["clip_id"] for r in self.records]==plan["selections"][split]
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i];x,receipt=read_cache(r,self.context)
        return dict(features=torch.from_numpy(x.copy()),length=len(x),real_length=r["effective_length"],target=torch.tensor(r["target_119"],dtype=torch.float32),
                    clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],split=self.split,frame_indices=r["frame_indices"],metadata=r,
                    tubelet_frame_indices=receipt["tubelet_frame_indices"],tubelet_real_counts=receipt["tubelet_real_counts"])


__all__=["CachedFeatureDataset","collate_features"]
