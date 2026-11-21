"""Real online/API/cache agreement, using one known Train video; never Test."""
from pathlib import Path
from fractions import Fraction
import json,shutil,time
import numpy as np
import torch
from urllib.request import urlopen,Request
from urllib.parse import urlencode
from .registry import ROOT,OUT,read_json,model_info
from .schema import validate_prediction
from animalact2.phase2_2.heads import VideoMAEProposalHead

def main():
    plan=read_json(ROOT/"animalact2/phase2_2/pilot_plan.json")
    r=plan["records"]["train"][0]
    source=ROOT/r["video_path"]
    fixture=OUT/"acceptance/fixtures/real_train_upload.mp4";fixture.parent.mkdir(parents=True,exist_ok=True)
    if not fixture.exists():shutil.copyfile(source,fixture)
    report=dict(source_split="train",source_video=str(source.relative_to(ROOT)),clip_id=r["clip_id"],
        standalone_fixture=str(fixture.relative_to(ROOT)),unseen_video=False,Test_read=False,checks=[])
    data=fixture.read_bytes()
    for key in ("attention","bigru_mean"):
        q=urlencode(dict(model=key,device="cuda",start=float(Fraction(r["effective_start_exact"])),end=float(Fraction(r["effective_end_exact"]))))
        with urlopen(Request("http://127.0.0.1:8765/api/predict?"+q,data=data,headers={"X-Filename":"real_train_upload.mp4","Content-Type":"application/octet-stream"}),timeout=300) as response:
            actual=validate_prediction(json.load(response))
        (OUT/f"acceptance/api_{key}.json").write_text(json.dumps(actual,indent=2),encoding="utf-8")
        assert actual["video"]["sampled_frame_indices"]==r["frame_indices"]
        cache=ROOT/f"animalact2/phase2_3_full_cache/cache/train/{r['clip_id']}/features.npy"
        x=torch.from_numpy(np.load(cache,allow_pickle=False)).unsqueeze(0)
        info=model_info(key); ck=torch.load(info["checkpoint"],map_location="cpu",weights_only=True)
        head=VideoMAEProposalHead(info["head"]);head.load_state_dict(ck["head_state"],strict=True);head.eval()
        with torch.inference_mode(): expected=head(x,torch.tensor([x.shape[1]]),torch.ones(x.shape[:2],dtype=torch.bool))[0]
        error=(expected-torch.tensor(actual["logits"])).abs().max().item()
        assert error<1e-4,(key,error)
        report["checks"].append(dict(model=key,cached_vs_online_max_logit_error=error,labels=actual["predicted_labels"],
            request_seconds=actual["request_seconds"],inference_seconds=actual["inference_time"],feature_shape=actual["feature_shape"],threshold=actual["threshold"]))
        print(report["checks"][-1],flush=True)
    report["passed"]=True
    (OUT/"acceptance/inference_agreement.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
