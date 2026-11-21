"""Check all three unchanged heads reading the same cache; forward-only, no CNN inference."""
import io
import json
import unittest
import torch
from animalact2.phase1_5.config import load_config
from animalact2.phase1_5.models import build_model
from animalact2.phase1_5_1.collate import move_batch_to_device
from .dataset import CachedFeatureDataset,collate_features,FeatureForwardView
from .support import ROOT,OUT,enable_guard,write_json
from . import tests


def main():
    enable_guard();torch.set_num_threads(4)
    stream=io.StringIO();tested=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    (OUT/"interface_test_results.txt").write_text(stream.getvalue(),encoding="utf-8")
    if not tested.wasSuccessful():print(stream.getvalue());raise RuntimeError("Interface tests failed")
    samples=[]
    for split in ("train","valid"):
        d=CachedFeatureDataset(split)
        for n in (1,10,16):
            i=next(i for i,r in enumerate(d.records) if r["effective_length"]==n);samples.append(d[i])
    batch=move_batch_to_device(collate_features(samples),"cuda");checks=[]
    for name in ("model1","model2","model3"):
        config=load_config(ROOT/f"animalact2/phase1_5/configs/{name}_seed42_bce.json")
        model=build_model(config).to("cuda").eval()
        def no_CNN(*args):raise AssertionError("Reader check must not run the CNN again")
        hook=model.backbone.encoder.register_forward_pre_hook(no_CNN)
        with torch.no_grad():
            result=model.forward_features(batch["frames"],batch["lengths"],batch["valid_frame_mask"],return_aux=True)
            via_view=FeatureForwardView(model)(batch["frames"],batch["lengths"],batch["valid_frame_mask"])
        assert result["logits"].shape==(6,119) and torch.isfinite(result["logits"]).all()
        torch.testing.assert_close(result["logits"],via_view,atol=0,rtol=0)
        if name=="model3":
            w=result["temporal_attention"]
            assert torch.all(w.masked_select(batch["padding_mask"][:,None,:].expand_as(w))==0)
            torch.testing.assert_close(w.sum(-1),torch.ones(6,119,device="cuda"),atol=1e-6,rtol=0)
        assert not model.backbone.encoder.training and all(p.grad is None for p in model.parameters())
        checks.append(dict(model=name,input_shape=list(batch["frames"].shape),lengths=batch["lengths"].tolist(),logits_shape=[6,119],finite=True,
                           same_cached_input=True,CNN_forward_calls=0,backward=False,parameters_updated=False))
        hook.remove();del model,result,via_view;torch.cuda.empty_cache()
    write_json(OUT/"reader_interface_checks.json",dict(status="passed",automated_tests=tested.testsRun,models=checks,Test_read=False,training_started=False,Checkpoint_modified=False))
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
