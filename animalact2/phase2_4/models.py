"""The unchanged Pilot heads plus a parameter-free frozen-cache reference."""
import copy
import json
import torch
from torch import nn
from animalact2.phase2_2.heads import VideoMAEProposalHead
from animalact2.phase1_5.config import load_config as original_config,seed_everything
from .support import ROOT,OUT,FULL,write_json

KINDS=dict(model1="mean",model2="bigru_mean",model3="bigru_attention")


def expected_config(name):
    c=copy.deepcopy(original_config(ROOT/f"animalact2/phase1_5/configs/{name}_seed42_bce.json"))
    ctx=json.loads((FULL/"cache/context.json").read_text(encoding="utf-8"))
    c["backbone"]=dict(weights=ctx["source"]["repo"],revision=ctx["source"]["revision"],frozen=True,encoder_loaded_in_training=False)
    c["input_cache"]=dict(root="animalact2/phase2_3_full_cache",feature_dim=768,dtype="float32",time_length="ceil(real_length/2)")
    return c


def load_config(path):
    c=json.loads(path.read_text(encoding="utf-8"))
    if c!=expected_config(c["model"]):raise ValueError("Only the approved first-round configuration is supported")
    return c


class NoEncoder(nn.Module):
    def forward(self,*args,**kwargs):raise RuntimeError("Encoder forward is forbidden during cached training")


class FrozenCacheReference(nn.Module):
    def __init__(self):
        super().__init__();ctx=json.loads((FULL/"cache/context.json").read_text(encoding="utf-8"))
        self.weight_info=dict(repo=ctx["source"]["repo"],revision=ctx["source"]["revision"],sha256=ctx["source"]["files"]["model.safetensors"],
                              role="frozen pretrained encoder reference; no encoder tensors instantiated")
        self.encoder=NoEncoder()
    def train(self,mode=True):return super().train(False)


class CachedVideoMAEHead(VideoMAEProposalHead):
    def __init__(self,kind):
        super().__init__(kind)
        self.backbone=FrozenCacheReference()
    def forward_features(self,features,lengths,valid_frame_mask,*,return_aux=False):
        return self.forward(features,lengths,valid_frame_mask,return_aux=return_aux)


def build_model(config):
    if config!=expected_config(config["model"]):raise ValueError("Locked configuration mismatch")
    seed_everything(config["seed"])
    return CachedVideoMAEHead(KINDS[config["model"]])


def prepare_configs():
    for name in KINDS:
        c=expected_config(name);path=OUT/f"configs/{name}_seed42_bce.json"
        if path.exists():assert json.loads(path.read_text(encoding="utf-8"))==c
        else:write_json(path,c)
