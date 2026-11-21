"""Run exactly the three approved formal configurations, sequentially and resumably."""
import gc
import json
import argparse
import time
import torch
from animalact2.phase1_5.config import config_hash
from .models import load_config
from .support import ROOT,OUT,enable_guard,write_json
from .trainer import FullCacheTrainer
from . import checkpoint


def main():
    enable_guard();torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute-training",action="store_true")
    args=parser.parse_args()
    if not args.execute_training:print("Preflight only. Formal run requires --execute-training.");return
    gate=json.loads((OUT/"integration_validation.json").read_text(encoding="utf-8"))
    assert gate["status"]=="passed" and gate["integration_fingerprints"]==checkpoint.fingerprints()
    states={};started=time.perf_counter()
    for name in ("model1","model2","model3"):
        config=load_config(OUT/f"configs/{name}_seed42_bce.json")
        directory=OUT/"runs"/f"{name}_bce_seed42";result_file=directory/"results.json"
        if result_file.exists():
            try:result=json.loads(result_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:result=None
            if result and result["status"]=="completed":
                assert result["config_SHA256"]==config_hash(config) and result["cache_provenance"]==checkpoint.cache_provenance()
                states[name]=dict(status="completed_previously",best_epoch=result["best_epoch"]);print(json.dumps(dict(model=name,skipped_completed=True)),flush=True);continue
        resume=directory/"last.pt" if (directory/"last.pt").exists() else None
        if directory.exists() and resume is None:raise RuntimeError("Incomplete experiment has no last checkpoint; do not overwrite: "+str(directory))
        print(json.dumps(dict(model=name,starting=True,resume=str(resume) if resume else None)),flush=True)
        trainer=FullCacheTrainer(config,directory)
        try:result=trainer.run(resume=resume,execute_training=True)
        finally:trainer.close()
        states[name]=dict(status="completed",best_epoch=result["best_epoch"],completed_epochs=result["completed_epochs"])
        write_json(OUT/"run_status.json",dict(models=states,Test_used=False,all_complete=len(states)==3))
        del trainer,result;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/"run_status.json",dict(models=states,all_complete=True,runner_seconds=time.perf_counter()-started,Test_used=False,extra_experiments_started=False))


if __name__=="__main__":main()
