"""Model 3: packed BiGRU + per-class masked temporal attention and classifier."""
import torch
from torch import nn
from .backbone import ResNet18Backbone
from .common import PackedBiGRU


class Model3(nn.Module):
    def __init__(self, *, backbone=None, hidden_size=128):
        super().__init__()
        self.backbone = backbone if backbone is not None else ResNet18Backbone()
        self.temporal = PackedBiGRU(hidden_size)
        self.attention = nn.Linear(2 * hidden_size, 119, bias=False)
        self.class_weight = nn.Parameter(torch.empty(119, 2 * hidden_size))
        self.class_bias = nn.Parameter(torch.zeros(119))
        nn.init.xavier_uniform_(self.class_weight)

    def forward_features(self, features, lengths, valid_frame_mask, *, return_aux=False):
        encoded = self.temporal(features, lengths, valid_frame_mask)
        scores = self.attention(encoded).transpose(1, 2).float()  # [B,119,T]
        weights = scores.masked_fill(~valid_frame_mask[:, None, :], float("-inf")).softmax(dim=-1)
        pooled = torch.bmm(weights.to(encoded.dtype), encoded)  # [B,119,2H]
        logits = (pooled * self.class_weight[None, :, :]).sum(dim=-1) + self.class_bias
        return dict(logits=logits, features=features, temporal_features=encoded, temporal_attention=weights, pooled=pooled) if return_aux else logits

    def forward(self, frames, lengths, valid_frame_mask, *, return_aux=False):
        features = self.backbone(frames, lengths, valid_frame_mask)
        return self.forward_features(features, lengths, valid_frame_mask, return_aux=return_aux)
