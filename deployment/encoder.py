"""Frozen VideoMAE, rectangular patches and real-length tubelets; weights unchanged."""
from __future__ import annotations
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from .artifacts import resolved_source,sha


def interpolate_positions(base,lt,hp,wp):
    """Original time-major [8,14,14,D]; spatial bicubic, temporal linear, no axis swap."""
    if base.ndim!=3 or base.shape[:2]!=(1,1568):raise ValueError("Expected native positional table")
    if min(lt,hp,wp)<1:raise ValueError("Empty grid")
    if (lt,hp,wp)==(8,14,14):return base
    dim=base.shape[-1]
    spatial=base.reshape(8,14,14,dim).permute(0,3,1,2)
    if (hp,wp)!=(14,14):spatial=F.interpolate(spatial,size=(hp,wp),mode="bicubic",align_corners=False)
    if lt!=8:
        time=spatial.permute(2,3,1,0).reshape(hp*wp,dim,8)
        spatial=F.interpolate(time,size=lt,mode="linear",align_corners=False).reshape(hp,wp,dim,lt).permute(3,2,0,1)
    return spatial.permute(0,2,3,1).reshape(1,lt*hp*wp,dim).contiguous()


def align_pixels(pixels):
    if pixels.ndim!=5 or pixels.shape[2]!=3 or not 1<=pixels.shape[1]<=16 or not torch.isfinite(pixels).all():
        raise ValueError("Expected finite real RGB-normalized [B,L,3,H,W], 1<=L<=16")
    b,l,c,h,w=pixels.shape
    pad_t=l%2;pad_h=(-h)%16;pad_w=(-w)%16
    # These are normalized zero values, not repeated frames or outside-window pixels.
    aligned=F.pad(pixels.permute(0,2,1,3,4),(0,pad_w,0,pad_h,0,pad_t),value=0.)
    assert torch.equal(aligned[:,:,:l,:h,:w],pixels.permute(0,2,1,3,4))
    return aligned,dict(real_length=l,aligned_length=l+pad_t,temporal_zero_slots=pad_t,real_hw=[h,w],
                       aligned_hw=[h+pad_h,w+pad_w],spatial_bottom_right_zero=[pad_h,pad_w],
                       grid=[(l+1)//2,(h+pad_h)//16,(w+pad_w)//16],
                       tubelet_real_counts=[min(2,l-i) for i in range(0,l,2)])


def encoder_state_hash(encoder):
    h=hashlib.sha256()
    for name,tensor in sorted(encoder.state_dict().items()):
        a=tensor.detach().cpu().contiguous();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    h.update(encoder.embeddings.position_embeddings.cpu().numpy().tobytes())
    return h.hexdigest()


class FrozenVideoMAEAdapter(nn.Module):
    def __init__(self,device="cuda"):
        super().__init__()
        from pathlib import Path
        from transformers import VideoMAEModel
        self.source=resolved_source();path=Path(self.source["path"])
        for name,expected in self.source["files"].items():
            if sha(path/name)!=expected:raise ValueError("Local pretrained file changed: "+name)
        self.encoder,info=VideoMAEModel.from_pretrained(path,local_files_only=True,attn_implementation="sdpa",output_loading_info=True)
        if info["missing_keys"] or info.get("mismatched_keys") or set(info["unexpected_keys"])-{"classifier.weight","classifier.bias","fc_norm.weight","fc_norm.bias"}:
            raise ValueError("Pretrained encoder did not load completely")
        self.loading_info=info
        cfg=self.encoder.config
        assert (cfg.num_frames,cfg.image_size,cfg.patch_size,cfg.tubelet_size,cfg.hidden_size)==(16,224,16,2,768)
        self.encoder.requires_grad_(False).eval().to(device)
        self.position_cache={}
        self.eval()

    def train(self,mode=True):
        super().train(False)
        if hasattr(self,"encoder"):self.encoder.eval()
        return self

    @torch.inference_mode()
    def patch_tokens(self,pixels):
        aligned,geometry=align_pixels(pixels)
        tokens=self.encoder.embeddings.patch_embeddings.projection(aligned).flatten(2).transpose(1,2)
        assert tokens.shape==(len(pixels),geometry["grid"][0]*geometry["grid"][1]*geometry["grid"][2],768)
        return tokens,geometry

    @torch.inference_mode()
    def forward(self,pixels,*,return_tokens=False):
        if self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):raise RuntimeError("Encoder must remain frozen/eval")
        tokens,geometry=self.patch_tokens(pixels)
        key=(*geometry["grid"],str(tokens.device),str(tokens.dtype))
        if key not in self.position_cache:
            self.position_cache[key]=interpolate_positions(self.encoder.embeddings.position_embeddings,*geometry["grid"]).to(tokens.device,tokens.dtype)
        x=tokens+self.position_cache[key]
        x=self.encoder.encoder(x).last_hidden_state
        if self.encoder.layernorm is not None:x=self.encoder.layernorm(x)
        lt,hp,wp=geometry["grid"]
        sequence=x.reshape(len(x),lt,hp*wp,768).mean(dim=2)
        assert sequence.shape==(len(x),lt,768) and torch.isfinite(sequence).all()
        result=dict(features=sequence,geometry=geometry)
        if return_tokens:result["tokens"]=x
        return result
