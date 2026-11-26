"""Official ImageNet ResNet-18 only; no download or AnimalAct task checkpoint loading."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from .common import validate_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / ".cache/torch/checkpoints/resnet18-f37072fd.pth"
OFFICIAL_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


class ResNet18Backbone(nn.Module):
    feature_dim = 512

    def __init__(self, *, weights_path=DEFAULT_WEIGHTS, frozen=True, frame_chunk_size=8):
        super().__init__()
        if not isinstance(frame_chunk_size, int) or frame_chunk_size < 1:
            raise ValueError("Frame chunk size must be positive")
        path = Path(weights_path).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
            raise FileNotFoundError("Requires the official local project weight file; no implicit downloads")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != OFFICIAL_SHA256:
            raise ValueError("Not the verified official ImageNet1K_V1 ResNet-18 weights")
        self.encoder = resnet18(weights=None)
        self.encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.encoder.fc = nn.Identity()
        self.weight_info = dict(name="ResNet18_Weights.IMAGENET1K_V1", official_url=ResNet18_Weights.IMAGENET1K_V1.url,
                                local_path=str(path), sha256=digest, AnimalAct_task_checkpoint=False)
        self.frame_chunk_size = frame_chunk_size
        self.set_frozen(frozen)

    def set_frozen(self, frozen=True):
        self.frozen = bool(frozen)
        self.encoder.requires_grad_(not self.frozen)
        self.encoder.train(self.training and not self.frozen)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, frames, lengths, valid_frame_mask):
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("Expected normalized RGB frames [B,T,3,H,W]")
        validate_sequence(frames, lengths, valid_frame_mask)
        if self.frozen:
            self.encoder.eval()
        # Invalid storage slots are not passed through the CNN or its BatchNorm.
        real = frames[valid_frame_mask]
        chunks = []
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            for chunk in real.split(self.frame_chunk_size):
                chunks.append(self.encoder(chunk))
        encoded = torch.cat(chunks)
        if encoded.shape != (real.shape[0], 512):
            raise RuntimeError("Unexpected backbone output")
        b, t = frames.shape[:2]
        flat_positions = valid_frame_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        features = encoded.new_zeros((b * t, 512)).index_copy(0, flat_positions, encoded)
        return features.view(b, t, 512)
