"""Model 2: ResNet-18 -> length-aware BiGRU -> masked mean -> 119 logits."""
from torch import nn
from .backbone import ResNet18Backbone
from .common import PackedBiGRU, masked_mean


class Model2(nn.Module):
    def __init__(self, *, backbone=None, hidden_size=128):
        super().__init__()
        self.backbone = backbone if backbone is not None else ResNet18Backbone()
        self.temporal = PackedBiGRU(hidden_size)
        self.classifier = nn.Linear(2 * hidden_size, 119)

    def forward_features(self, features, lengths, valid_frame_mask, *, return_aux=False):
        encoded = self.temporal(features, lengths, valid_frame_mask)
        pooled = masked_mean(encoded, valid_frame_mask)
        logits = self.classifier(pooled)
        return dict(logits=logits, features=features, temporal_features=encoded, pooled=pooled) if return_aux else logits

    def forward(self, frames, lengths, valid_frame_mask, *, return_aux=False):
        features = self.backbone(frames, lengths, valid_frame_mask)
        return self.forward_features(features, lengths, valid_frame_mask, return_aux=return_aux)
