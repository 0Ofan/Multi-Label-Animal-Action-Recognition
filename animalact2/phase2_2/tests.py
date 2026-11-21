"""Synthetic native-equivalence/geometry tests and real-cache head gradient tests."""
import gc
import json
import time
import torch
from animalact2.phase1_5.losses import MultilabelLoss,UnsupportedClassPolicy
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from .adapter import align_pixels,interpolate_positions
from .data import CachedPilotDataset,collate_features
from .heads import VideoMAEProposalHead
from .support import OUT,write_json


@torch.inference_mode()
def validate_adapter(adapter):
    started=time.perf_counter();torch.manual_seed(42)
    x=torch.randn(1,16,3,224,224,device="cuda")
    original=adapter.encoder(pixel_values=x,bool_masked_pos=None).last_hidden_state
    changed=adapter(x,return_tokens=True)
    error=float((original-changed["tokens"]).abs().max())
    torch.testing.assert_close(original,changed["tokens"],atol=0,rtol=0)
    torch.testing.assert_close(original.reshape(1,8,196,768).mean(2),changed["features"],atol=0,rtol=0)
    del x,original,changed
    # Explicit coordinate-valued fields detect time/height/width permutation errors.
    t,y,x=torch.meshgrid(torch.arange(8.),torch.arange(14.),torch.arange(14.),indexing="ij")
    base=torch.stack([t,y,x],-1).reshape(1,1568,3)
    p=interpolate_positions(base,3,14,25).reshape(3,14,25,3)
    torch.testing.assert_close(p[...,0],p[:,0,0,0][:,None,None].expand(3,14,25),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(p[...,1],p[0,:,0,1][None,:,None].expand(3,14,25),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(p[...,2],p[0,0,:,2][None,None,:].expand(3,14,25),atol=1e-5,rtol=1e-5)
    assert torch.all(p[1:,0,0,0]>p[:-1,0,0,0]) and torch.all(p[0,1:,0,1]>p[0,:-1,0,1]) and torch.all(p[0,0,1:,2]>p[0,0,:-1,2])
    assert torch.equal(interpolate_positions(base,8,14,14),base)
    # Every possible real frame position, including the final odd/single frame,
    # must influence exactly its own tubelet's bottom-right patch before attention.
    coverage=[]
    for length in range(1,17):
        real=torch.zeros(1,length,3,224,398,device="cuda")
        baseline,geometry=adapter.patch_tokens(real)
        aligned,_=align_pixels(real)
        assert geometry["aligned_hw"]==[224,400] and geometry["aligned_length"]==length+(length%2)
        assert aligned.shape==(1,3,length+(length%2),224,400)
        assert torch.all(aligned[:,:,:,-1,-2:]==0)
        del aligned
        for frame in range(length):
            probe=real.clone();probe[0,frame,:,-1,-1]=1.
            tokens,_=adapter.patch_tokens(probe)
            nonzero=(tokens-baseline).abs().amax(-1)[0].nonzero().flatten().tolist()
            expected=(frame//2)*350+13*25+24
            assert nonzero==[expected],(length,frame,nonzero,expected)
            coverage.append(dict(real_length=length,frame_position=frame,changed_patch_index=expected))
        del real,baseline,probe,tokens
    result=dict(status="passed",native_16x224x224_max_token_error=error,position_time_height_width_orientation=True,
                native_position_table_unchanged=True,synthetic_real_frame_and_right_edge_probes=len(coverage),
                lengths_tested=list(range(1,17)),all_real_frames_have_patch_support=True,rightmost_real_column_retained=True,
                odd_and_single_frame_retained=True,details=coverage,seconds=time.perf_counter()-started)
    write_json(OUT/"adapter_validation.json",result)
    gc.collect();torch.cuda.empty_cache()
    return result


def validate_heads():
    train,valid=CachedPilotDataset("train"),CachedPilotDataset("valid")
    def pick(ds,desired):
        return [ds[next(i for i,r in enumerate(ds.records) if r["effective_length"]==n)] for n in desired]
    # Deliberately unsorted and mixed true lengths; gradients use Train only.
    tb=collate_features(pick(train,[1,16,3,5,7,15,2,4]));vb=collate_features(pick(valid,[1,3,15,16]))
    assert tb["real_lengths"]==[1,16,3,5,7,15,2,4] and tb["lengths"].tolist()==[1,8,2,3,4,8,1,2]
    counts=torch.tensor([r["target_119"] for r in ReleaseClipDataset("train").records]).sum(0)
    assert (counts==0).nonzero().flatten().tolist()==[23]
    checks=[];torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for kind in ("mean","bigru_mean","bigru_attention"):
        torch.manual_seed(42);head=VideoMAEProposalHead(kind).eval()
        with torch.no_grad():cpu=head(tb["features"],tb["lengths"],tb["valid_frame_mask"])
        head=head.cuda().eval();policy=UnsupportedClassPolicy(head,counts)
        x=tb["features"].cuda();mask=tb["valid_frame_mask"].cuda()
        with torch.no_grad():
            gpu=head(x,tb["lengths"],mask,return_aux=True);poison=x.clone();poison[~mask]=12345.
            unchanged=head(poison,tb["lengths"],mask)
            pad_error=float((gpu["logits"]-unchanged).abs().max())
            assert pad_error==0 and gpu["logits"].shape==(8,119) and torch.isfinite(gpu["logits"]).all()
            cpu_error=float((cpu-gpu["logits"].cpu()).abs().max())
            torch.testing.assert_close(cpu,gpu["logits"].cpu(),atol=1e-4,rtol=1e-4)
            vz=head(vb["features"].cuda(),vb["lengths"],vb["valid_frame_mask"].cuda())
            assert vz.shape==(4,119) and torch.isfinite(vz).all()
            if kind=="bigru_attention":
                a=gpu["temporal_attention"]
                assert torch.all(a.masked_select(~mask[:,None,:])==0)
                torch.testing.assert_close(a.sum(-1),torch.ones(8,119,device="cuda"))
        head.train();head.zero_grad(set_to_none=True)
        criterion=MultilabelLoss(dict(name="bce",gamma_neg=4.,gamma_pos=0.,clip=.05),policy.active_mask).cuda()
        output=head(x,tb["lengths"],mask);loss=criterion(output,tb["target"].cuda());assert torch.isfinite(loss)
        loss.backward()
        norms={name:float(p.grad.norm()) if p.grad is not None else None for name,p in head.named_parameters()}
        assert all(v is not None and v>0 for v in norms.values())
        assert all(torch.isfinite(p.grad).all() for p in head.parameters())
        for name in policy.anchors:assert torch.all(dict(head.named_parameters())[name].grad[23]==0)
        checks.append(dict(head=kind,status="passed",parameters=sum(p.numel() for p in head.parameters()),
                           Train_batch=8,Valid_batch=4,output_shape=[8,119],Train_real_lengths=tb["real_lengths"],
                           effective_tubelet_lengths=tb["lengths"].tolist(),CPU_CUDA_max_logit_error=cpu_error,
                           padded_feature_perturbation_max_logit_error=pad_error,loss=float(loss.detach()),
                           gradient_norms=norms,unsupported_class_loss_gradient_zero=True,optimizer_updates=0))
        del head,x,mask,gpu,unchanged,poison,vz,cpu,output,loss,criterion,policy;gc.collect();torch.cuda.empty_cache()
    result=dict(status="passed",heads=checks,diagnostic_backward_calls=3,optimizer_updates=0,
                CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                formal_training=False,Valid_used_for_gradients=False,Test_read=False)
    write_json(OUT/"head_validation.json",result)
    return result
