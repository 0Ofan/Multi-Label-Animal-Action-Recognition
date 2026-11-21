"""Release-v1 full field of view, aspect-preserving PIL bilinear resize."""
from __future__ import annotations
import numpy as np
from PIL import Image
import torch

PREPROCESSING_ID = "rgb_short_edge_224_bilinear_no_crop_imagenet_v1"
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


class FullFrameTransform:
    def __init__(self):
        self.mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, rgb: np.ndarray) -> torch.Tensor:
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an actual uint8 RGB frame [H,W,3]")
        h, w = rgb.shape[:2]
        if min(h, w) < 1:
            raise ValueError("Empty frame")
        size = (int(224 * w / h), 224) if h <= w else (224, int(224 * h / w))
        resized = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).contiguous().float().div_(255)
        return (tensor - self.mean) / self.std
