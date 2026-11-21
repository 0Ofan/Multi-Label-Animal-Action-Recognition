"""Exactly 100 Train / 20 Valid Clip feature extraction and bounded diagnostics."""
from __future__ import annotations
import csv
import gc
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from animalact2.phase1_5_1.transforms import MEAN,STD
from .support import ROOT,OUT,enable_guard,write_json,sha,RAMMonitor
from .adapter import FrozenVideoMAEAdapter,encoder_state_hash,align_pixels
from .data import get_plan,cache_context,read_cache,commit,FORCED
from .tests import validate_adapter,validate_heads


def make_visuals(plan):
    """Selected pilot media only. The display marks padding, never changes model pixels."""
    font_path=Path("C:/Windows/Fonts/arial.ttf")
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    panels=[];records=[]
    for split in ("train","valid"):
        ids=[r["clip_id"] for r in plan["records"][split] if r["clip_id"] in FORCED[split]]
        ds=ReleaseClipDataset(split,clip_ids=ids)
        for i,r in enumerate(ds.records):
            item=ds[i];k=len(item["frames"])-1
            frame=item["frames"][k]
            rgb=frame*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]
            rgb=rgb.clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()
            aligned,g=align_pixels(item["frames"][None])
            assert torch.equal(aligned[0,:,:len(item["frames"]),:224,:398],item["frames"].permute(1,0,2,3))
            original=Image.fromarray(rgb)
            padded=Image.new("RGB",(400,224),tuple(round(v*255) for v in MEAN));padded.paste(original,(0,0))
            # Magenta only in the review image to make the 2-column synthetic area visible.
            draw=ImageDraw.Draw(padded);draw.rectangle((398,0,399,223),fill=(255,0,255))
            panel=Image.new("RGB",(840,285),"white");d=ImageDraw.Draw(panel)
            d.text((10,5),f"{split} {r['parent_video_id']} | L={r['effective_length']} | real frame {r['frame_indices'][k]}",fill="black",font=font)
            d.text((10,29),"Full 398x224 RGB",fill="black",font=font)
            d.text((430,29),"400x224 patch alignment: right 2 columns marked",fill="black",font=font)
            panel.paste(original,(10,53));panel.paste(padded,(430,53));panels.append(panel)
            p=OUT/"visualizations"/f"{r['clip_id']}.jpg";p.parent.mkdir(parents=True,exist_ok=True);panel.save(p,quality=95)
            records.append(dict(clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],split=split,display_frame_index=r["frame_indices"][k],
                                reason=FORCED[split][r["clip_id"]],image=p.relative_to(OUT).as_posix(),
                                scope="full resized real frame vs internal patch alignment; magenta is review markup only"))
    sheet=Image.new("RGB",(840,len(panels)*285),"white")
    for i,p in enumerate(panels):sheet.paste(p,(0,i*285))
    sheet.save(OUT/"visualizations/geometry_contact_sheet.jpg",quality=94)
    write_json(OUT/"visualizations/index.json",records)


def extract(adapter,plan,ctx):
    if (OUT/"cache/context.json").exists():assert json.loads((OUT/"cache/context.json").read_text())==ctx
    else:write_json(OUT/"cache/context.json",ctx)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();success={"train":0,"valid":0};reused=0;new=0;checks=[];batch_times=[]
    with RAMMonitor() as ram:
        for split in ("train","valid"):
            ds=ReleaseClipDataset(split,clip_ids=[r["clip_id"] for r in plan["records"][split]])
            for length in range(1,17):
                pending=[]
                for i,r in enumerate(ds.records):
                    if r["effective_length"]!=length:continue
                    cached=read_cache(r,ctx)
                    if cached is not None:reused+=1;success[split]+=1
                    else:pending.append(i)
                for start in range(0,len(pending),2):
                    begin=time.perf_counter();indices=pending[start:start+2];items=[ds[i] for i in indices]
                    pixels=torch.stack([item["frames"] for item in items]).cuda()
                    assert pixels.shape==(len(items),length,3,224,398)
                    result=adapter(pixels);features=result["features"].float().cpu().numpy().copy()
                    for j,item in enumerate(items):
                        r=ds.records[indices[j]]
                        assert item["frame_indices"].tolist()==r["frame_indices"] and item["effective_length"]==length
                        assert item["target"].tolist()==r["target_119"] and item["label_sources"]==r["label_evidence"]
                        commit(r,ctx,np.ascontiguousarray(features[j]),result["geometry"])
                        success[split]+=1;new+=1
                    torch.cuda.synchronize();batch_times.append(dict(split=split,real_length=length,batch=len(items),seconds=time.perf_counter()-begin))
                    del pixels,result,features,items
                print(json.dumps(dict(split=split,real_length=length,completed=success[split],cache_reused=reused)),flush=True)
        assert success=={"train":100,"valid":20}
        # Independent receipt/array/Manifest correspondence checks for all 120 records.
        for split,records in plan["records"].items():
            for r in records:
                x,receipt=read_cache(r,ctx)
                assert receipt["geometry"]["grid"]==[len(x),14,25]
                assert sum(receipt["tubelet_real_counts"])==r["effective_length"]
                checks.append(dict(split=split,clip_id=r["clip_id"],parent_video_id=r["parent_video_id"],real_length=r["effective_length"],
                                   feature_length=len(x),feature_dim=x.shape[1],label_count=len(r["class_ids"]),
                                   real_frames_preserved=True,full_field_pixels_preserved=True,frame_label_source_alignment=True,
                                   temporal_zero_slots=r["effective_length"]%2,numeric_finite=True,feature_SHA256=receipt["features_SHA256"]))
        torch.cuda.synchronize()
        result=dict(status="passed",counts=success,failed=0,new_extractions=new,reused=reused,
                    seconds=time.perf_counter()-started,batch_size_max=2,same_real_length_buckets=True,precision="fp32",attention="sdpa",
                    real_frames=sum(r["effective_length"] for records in plan["records"].values() for r in records),
                    feature_time_steps=sum((r["effective_length"]+1)//2 for records in plan["records"].values() for r in records),
                    CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    RAM=ram.result(),batch_timings=batch_times,Test_read=False,formal_training=False)
    write_json(OUT/"extraction_summary.json",result)
    with (OUT/"clip_validation.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    return result


def main():
    enable_guard();torch.set_num_threads(4)
    if (OUT/"pilot_validation.json").exists():raise FileExistsError("Keep the completed pilot; do not overwrite")
    if not (OUT/"protection_baseline.json").exists():raise RuntimeError("Capture history protection baseline first")
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.set_float32_matmul_precision("highest");torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter()
    with RAMMonitor() as overall_ram:
        plan=get_plan();ctx=cache_context()
        adapter=FrozenVideoMAEAdapter();before=encoder_state_hash(adapter.encoder)
        print(json.dumps(dict(stage="adapter_tests",GPU=torch.cuda.get_device_name(0))),flush=True)
        adapter_checks=validate_adapter(adapter)
        print(json.dumps(dict(adapter_validation="passed",native_error=adapter_checks["native_16x224x224_max_token_error"])),flush=True)
        adapter.position_cache.clear();gc.collect();torch.cuda.empty_cache()
        extraction=extract(adapter,plan,ctx)
        assert encoder_state_hash(adapter.encoder)==before
        assert all(not p.requires_grad and p.grad is None for p in adapter.encoder.parameters()) and not adapter.encoder.training
        adapter.cpu();adapter.position_cache.clear();gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(stage="head_forward_backward_tests",extracted=sum(extraction["counts"].values()))),flush=True)
        heads=validate_heads()
        assert encoder_state_hash(adapter.encoder)==before
        assert all(not p.requires_grad and p.grad is None for p in adapter.encoder.parameters())
        make_visuals(plan)
        result=dict(status="passed",Train_clips=100,Valid_clips=20,feature_dim=768,temporal_length="ceil(real_length/2)",
                    encoder_parameters=sum(p.numel() for p in adapter.encoder.parameters()),encoder_state_before=before,encoder_state_after=encoder_state_hash(adapter.encoder),
                    encoder_frozen_eval=True,encoder_gradients=None,head_diagnostic_backwards=3,optimizer_updates=0,
                    source=adapter.source,loading_info=adapter.loading_info,program_seconds=time.perf_counter()-started,overall_RAM=overall_ram.result(),
                    python=platform.python_version(),interpreter=sys.executable,torch=str(torch.__version__),GPU=torch.cuda.get_device_name(0),
                    Test_read=False,full_cache_generated=False,formal_training=False,new_weights_downloaded=False)
    write_json(OUT/"pilot_validation.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
