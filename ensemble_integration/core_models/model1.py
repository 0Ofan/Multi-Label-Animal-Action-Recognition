"""Model 1: ResNet-18 -> masked temporal mean -> 119 independent logits."""
from torch import nn
from .backbone import ResNet18Backbone
from .common import masked_mean, validate_sequence


class Model1(nn.Module):
    def __init__(self, *, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else ResNet18Backbone()
        self.classifier = nn.Linear(512, 119)

    def forward_features(self, features, lengths, valid_frame_mask, *, return_aux=False):
        validate_sequence(features, lengths, valid_frame_mask)
        if features.shape[-1] != 512:
            raise ValueError("Expected ResNet-18 features")
        pooled = masked_mean(features, valid_frame_mask)
        logits = self.classifier(pooled)
        return dict(logits=logits, features=features, pooled=pooled) if return_aux else logits

    def forward(self, frames, lengths, valid_frame_mask, *, return_aux=False):
        features = self.backbone(frames, lengths, valid_frame_mask)
        return self.forward_features(features, lengths, valid_frame_mask, return_aux=return_aux)
