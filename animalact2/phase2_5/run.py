import gc,json,time
import torch
from .support import OUT,enable_guard,write_json
from .models import expected_config
from .trainer import FullCacheTrainer
from . import checkpoint

def main():
    enable_guard();torch.set_num_threads(4)
    states={};start=time.perf_counter()
    for seed in (123,2026):
        for name in ('model1','model2','model3'):
            key=f'{name}_seed{seed}';d=OUT/'runs'/key;c=expected_config(name,seed)
            write_json(OUT/f'configs/{key}.json',c)
            if (d/'results.json').exists():
                r=json.loads((d/'results.json').read_text());assert r['status']=='completed' and r['config']==c
            else:
                trainer=FullCacheTrainer(c,d)
                try:r=trainer.run(resume=d/'last.pt' if (d/'last.pt').exists() else None,execute_training=True)
                finally:trainer.close()
                del trainer;gc.collect();torch.cuda.empty_cache()
            states[key]=dict(best_epoch=r['best_epoch'],epochs=r['completed_epochs'])
            write_json(OUT/'run_status.json',dict(models=states,all_complete=len(states)==6,runner_seconds=time.perf_counter()-start,Test_used=False))
    print('all six runs complete',flush=True)
if __name__=='__main__':main()
