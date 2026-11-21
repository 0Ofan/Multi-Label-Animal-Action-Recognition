"""One fresh-process cached Model3 pilot epoch, to isolate RAM from prior RGB buffers."""
import gc
import time
import torch
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from animalact2.phase1_5.data import training_class_support
from animalact2.phase1_5.models import build_model
from animalact2.phase1_5.config import load_config
from animalact2.phase1_5.engine import make_optimizer,train_one_epoch,evaluate_loader
from animalact2.phase1_5.losses import MultilabelLoss,UnsupportedClassPolicy
from animalact2.phase1_5.schedule import LearningRatePlan
from .dataset import CachedFeatureDataset,FeatureForwardView,make_pilot_loader
from .support import ROOT,OUT,enable_guard,RAMMonitor,write_json


def main():
    enable_guard();torch.set_num_threads(4)
    if (OUT/"fresh_process_cached_resources.json").exists():raise FileExistsError("Preserve resource probe")
    with RAMMonitor() as ram:
        config=load_config(ROOT/"animalact2/phase1_5/configs/model3_seed42_bce.json")
        counts=training_class_support(ReleaseClipDataset("train"))
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=build_model(config).to("cuda");policy=UnsupportedClassPolicy(model,counts)
        criterion=MultilabelLoss(config["loss"],policy.active_mask).to("cuda");optimizer=make_optimizer(model,config)
        schedule=LearningRatePlan(optimizer,config);schedule.apply_epoch(1)
        train_loader=make_pilot_loader(CachedFeatureDataset("train"),config,training=True,cached=True)
        valid_loader=make_pilot_loader(CachedFeatureDataset("valid"),config,training=False,cached=True)
        start=time.perf_counter();a=next(iter(train_loader));b=next(iter(valid_loader));startup=time.perf_counter()-start
        del a,b
        torch.cuda.synchronize();start=time.perf_counter();view=FeatureForwardView(model)
        training=train_one_epoch(view,train_loader,optimizer,criterion,policy,"cuda",accumulation_steps=4,grad_clip_norm=5.)
        valid=evaluate_loader(view,valid_loader,criterion,"cuda");torch.cuda.synchronize();seconds=time.perf_counter()-start
        assert training["examples"]==100 and valid["examples"]==20 and training["optimizer_updates"]==4
        allocated,reserved=torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
        del model,view,train_loader,valid_loader,training,valid;gc.collect()
    result=dict(status="passed",model="model3",mode="cached_only_fresh_process",Train_clips=100,Valid_clips=20,
                no_raw_video_decode=True,optimizer_updates=4,worker_count_per_loader=2,worker_startup_seconds=startup,
                steady_pilot_epoch_seconds=seconds,CUDA_peak_allocated_bytes=allocated,CUDA_peak_reserved_bytes=reserved,
                RAM=ram.result(),formal_config_modified=False,formal_training=False,temporal_outputs_or_weights_saved=False,Test_read=False)
    write_json(OUT/"fresh_process_cached_resources.json",result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
