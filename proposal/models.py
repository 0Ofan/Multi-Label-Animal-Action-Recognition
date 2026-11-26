"""Reuse the proposal's existing ResNet classification heads."""

import torch
from torch import nn

from animalact2.phase1_5_1.model1 import Model1
from animalact2.phase1_5_1.model2 import Model2
from animalact2.phase1_5_1.model3 import Model3


KINDS = {"mean": Model1, "bigru_mean": Model2, "bigru_attention": Model3}


def build_model(kind):
    return KINDS[kind](backbone=nn.Identity())


def classify(model, features):
    lengths = torch.full((len(features),), features.shape[1], dtype=torch.int64)
    mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    return model.forward_features(features, lengths, mask)
