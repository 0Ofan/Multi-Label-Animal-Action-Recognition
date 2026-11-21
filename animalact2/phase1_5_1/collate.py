"""Dynamic temporal storage padding, never repeated frames or fixed T=16."""
from __future__ import annotations
import copy
import torch


def collate_clips(samples):
    if not samples:
        raise ValueError("Empty batch")
    if len({tuple(s["frames"].shape[1:]) for s in samples}) != 1:
        raise ValueError("Different aspect ratios require spatial bucketing; never silently crop/stretch")
    lengths = torch.tensor([s["effective_length"] for s in samples], dtype=torch.int64)
    if torch.any(lengths < 1):
        raise ValueError("Zero-length clip is not a usable sample")
    tmax = int(lengths.max())
    frames = samples[0]["frames"].new_zeros((len(samples), tmax, *samples[0]["frames"].shape[1:]))
    valid = torch.zeros(len(samples), tmax, dtype=torch.bool)
    indices = torch.full((len(samples), tmax), -1, dtype=torch.int64)
    for i, s in enumerate(samples):
        n = int(lengths[i])
        if (s["frames"].shape[0] != n or s["frame_indices"].shape != (n,)
                or not torch.equal(s["valid_frame_mask"], torch.ones(n, dtype=torch.bool))):
            raise ValueError("Length, mask, frames and frame indices disagree")
        if s["frames"].dtype != torch.float32 or not torch.isfinite(s["frames"]).all():
            raise ValueError("Expected normalized finite FP32 RGB")
        if s["target"].shape != (119,) or not torch.all((s["target"] == 0) | (s["target"] == 1)):
            raise ValueError("Invalid 119-dimensional multi-label target")
        frames[i, :n] = s["frames"]
        valid[i, :n] = True
        indices[i, :n] = s["frame_indices"]
    return dict(frames=frames, target=torch.stack([s["target"] for s in samples]), lengths=lengths,
                valid_frame_mask=valid, padding_mask=~valid, frame_indices=indices,
                clip_ids=[s["clip_id"] for s in samples], parent_video_ids=[s["parent_video_id"] for s in samples],
                splits=[s["split"] for s in samples], metadata=[copy.deepcopy(s["metadata"]) for s in samples],
                label_sources=[copy.deepcopy(s["label_sources"]) for s in samples])


def move_batch_to_device(batch, device):
    # pack_padded_sequence requires CPU lengths; mask/images share the model device.
    return {k: v.to(device) if isinstance(v, torch.Tensor) and k != "lengths" else v for k, v in batch.items()}
