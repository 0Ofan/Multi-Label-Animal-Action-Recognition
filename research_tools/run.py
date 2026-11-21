"""Portable 2.0 cached-head experiments. No Test and no historical overwrite."""
import argparse,json,time,copy,csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from common import bootstrap,CachedClips,collate,FeatureView,read,sha

def parser():
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,default=Path(__file__).resolve().parents[1]/"research_workspace")
    p.add_argument("--family",choices=["resnet18","videomae"],default="videomae");p.add_argument("--model",choices=["model1","model2","model3"],default="model3")
    a=p.add_mutually_exclusive_group(required=True);a.add_argument("--smoke",action="store_true");a.add_argument("--train",action="store_true");a.add_argument("--evaluate",type=Path)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto");p.add_argument("--seed",type=int,default=42)
    p.add_argument("--loss",choices=["bce","asl"],default="bce");p.add_argument("--gamma-neg",type=float,default=4.)
    p.add_argument("--lr",type=float);p.add_argument("--weight-decay",type=float);p.add_argument("--max-epochs",type=int);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--output",type=Path);p.add_argument("--resume",type=Path);p.add_argument("--initial-checkpoint",type=Path)
    return p

def main():
    a=parser().parse_args();root=bootstrap(a.workspace);torch.set_num_threads(4)
    from animalact2.phase1_5.config import seed_everything,config_hash,development_fingerprints
    from animalact2.phase1_5.data import EpochShuffleSampler
    from animalact2.phase1_5.losses import UnsupportedClassPolicy,MultilabelLoss
    from animalact2.phase1_5.engine import train_one_epoch,evaluate_loader,summarize,make_optimizer
    from animalact2.phase1_5.schedule import LearningRatePlan,EarlyStopping
    from animalact2.phase1_5.checkpoint import rng_state,restore_rng
    from animalact2.phase1_5.evaluation import select_validation_threshold
    device=("cuda" if torch.cuda.is_available() else "cpu") if a.device=="auto" else a.device
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    phase="first_round" if a.family=="resnet18" else "phase2_4"
    c=read(root/f"animalact2/{phase}/runs/{a.model}_bce_seed42/config.json")
    c["seed"]=a.seed;c["device"]=device;c["data"]["num_workers"]=0 if a.smoke else a.workers
    c["loss"]["name"]=a.loss;c["loss"]["gamma_neg"]=a.gamma_neg
    if a.lr is not None:c["optimizer"]["lr"]=a.lr
    if a.weight_decay is not None:c["optimizer"]["weight_decay"]=a.weight_decay
    if a.max_epochs is not None:c["training"]["max_epochs"]=a.max_epochs
    if min(c["optimizer"]["lr"],c["training"]["max_epochs"])<=0:raise ValueError("Invalid experiment settings")
    seed_everything(a.seed)
    if a.family=="resnet18":
        from animalact2.phase1_5.models import build_model
        model=build_model(c)
    else:
        from animalact2.phase2_2.heads import VideoMAEProposalHead
        model=VideoMAEProposalHead(dict(model1="mean",model2="bigru_mean",model3="bigru_attention")[a.model])
    training=CachedClips(root,a.family,"train");validation=CachedClips(root,a.family,"valid")
    counts=torch.tensor([r["target_119"] for r in training.records]).sum(0)
    def load_head(path):
        ck=torch.load(path,map_location="cpu",weights_only=True)
        if ck.get("diagnostic"):raise ValueError("Diagnostic checkpoints cannot be used as formal initialization/evaluation")
        if "config" in ck and ck["config"]["model"]!=a.model:raise ValueError("Different classification head")
        if "input_mode" in ck and ck["input_mode"]!=("shared_full_cache_fp32" if a.family=="resnet18" else "videomae_full_cache_fp32"):raise ValueError("Different backbone cache")
        if ck.get("data_fingerprints",development_fingerprints())!=development_fingerprints():raise ValueError("Different data release")
        result=model.load_state_dict(ck["head_state"],strict=False)
        if result.unexpected_keys or any(not k.startswith("backbone.") for k in result.missing_keys):raise ValueError("Different head architecture")
        return ck
    if a.evaluate:source_state=load_head(a.evaluate)
    if a.initial_checkpoint:load_head(a.initial_checkpoint)
    model.to(device);view=FeatureView(model);policy=UnsupportedClassPolicy(model,counts)
    assert policy.unsupported.tolist()==[23]
    criterion=MultilabelLoss(c["loss"],policy.active_mask).to(device);optimizer=make_optimizer(model,c)
    schedule=LearningRatePlan(optimizer,c);early=EarlyStopping(c["training"]["patience"],c["training"]["min_delta"])
    cache_fp={f:sha(training.folder/f) for f in ("cache/context.json","full_plan.json","validation_summary.json")}
    source_fp={"research_tools/"+name:sha(Path(__file__).parent/name) for name in ("run.py","common.py")}
    source_fp.update({str(p.relative_to(root)):sha(p) for folder in (root/"animalact2/phase1_5_1",root/"animalact2/phase2_2") for p in folder.glob("*.py")})
    if a.smoke:
        selected=[]
        for length in (1,2,3,4,5,10,15,16):
            selected.append(next(i for i,r in enumerate(training.records) if r["effective_length"]==length))
        tds=Subset(training,selected);vds=Subset(validation,range(32))
    else:tds=training;vds=validation
    sampler=EpochShuffleSampler(tds,a.seed)
    def loader(ds,train):
        return DataLoader(ds,batch_size=c["data"]["batch_size"] if train else c["data"]["valid_batch_size"],sampler=sampler if train else None,
            shuffle=False,collate_fn=collate,num_workers=c["data"]["num_workers"],pin_memory=device=="cuda",drop_last=False,
            generator=torch.Generator().manual_seed(a.seed+(0 if train else 17)))
    tl,vl=loader(tds,True),loader(vds,False)
    def emit(value):print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
    if a.evaluate:
        output=evaluate_loader(view,vl,criterion,device)
        threshold_path=a.evaluate.parent/"threshold_selection.json"
        threshold=read(threshold_path)["threshold"] if threshold_path.exists() and a.evaluate.name=="best.pt" else select_validation_threshold(output["logits"],output["target"],split="valid")["threshold"]
        result=summarize(output,policy,threshold=threshold)
        saved=a.evaluate.parent/"results.json"
        expected=read(saved).get("valid",{}).get("main_mAP") if saved.exists() else None
        result.update(source_checkpoint=str(a.evaluate),family=a.family,seed=source_state.get("config",{}).get("seed",a.seed),Test_read=False,
            expected_saved_mAP=expected,mAP_abs_error=None if expected is None else abs(result["main_mAP"]-expected))
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            if a.output.exists():raise FileExistsError(a.output)
            a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        emit({k:v for k,v in result.items() if k!="per_class"});return
    if a.smoke:
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        schedule.apply_epoch(1);sampler.set_epoch(1)
        t=train_one_epoch(view,tl,optimizer,criterion,policy,device,accumulation_steps=c["data"]["accumulation_steps"],grad_clip_norm=c["optimizer"]["grad_clip_norm"])
        v=evaluate_loader(view,vl,criterion,device)
        changed=sum(not torch.equal(before[n],p) for n,p in model.named_parameters() if n in before)
        assert changed>0 and t["optimizer_updates"]==1 and torch.isfinite(v["logits"]).all()
        result=dict(status="smoke_passed",family=a.family,model=a.model,Train_examples=t["examples"],Valid_examples=v["examples"],
            updated_parameter_tensors=changed,optimizer_updates=1,train_objective=t["objective"],valid_objective=v["objective"],Test_read=False,formal_training=False)
        if a.output:
            if a.output.exists():raise FileExistsError(a.output)
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
        emit(result);return
    if a.resume and a.initial_checkpoint:raise ValueError("Choose resume or weight initialization")
    out=(a.output if a.output else root/"new_experiments"/(a.family+"_"+a.model+"_"+str(a.seed)+"_"+str(time.time_ns()))).resolve()
    if not out.is_relative_to(root/"new_experiments"):raise ValueError("New runs must stay inside workspace/new_experiments")
    history=[];start=1
    if a.resume:
        if a.resume.resolve().parent!=out:raise ValueError("Resume must belong to this new run")
        ck=torch.load(a.resume,map_location="cpu",weights_only=True)
        if ck.get("format")!="animalact2-research-tools-v1" or ck["family"]!=a.family or ck["config"]!=c or ck["cache_fingerprints"]!=cache_fp or ck.get("implementation_fingerprints")!=source_fp:raise ValueError("Resume provenance mismatch")
        load_head(a.resume);optimizer.load_state_dict(ck["optimizer"]);policy.load_state_dict(ck["class_policy"]);policy.restore_rows(model)
        schedule.load_state_dict(ck["schedule"]);early.load_state_dict(ck["early"]);restore_rng(ck["RNG"]);history=ck["history"];start=ck["epoch"]+1
    else:
        if out.exists():raise FileExistsError("Refuse to overwrite a run")
        out.mkdir(parents=True);(out/"config.json").write_text(json.dumps(c,indent=2),encoding="utf-8")
    for epoch in range(start,c["training"]["max_epochs"]+1):
        if early.bad_epochs>=early.patience:break
        began=time.perf_counter();schedule.apply_epoch(epoch);sampler.set_epoch(epoch)
        t=train_one_epoch(view,tl,optimizer,criterion,policy,device,accumulation_steps=c["data"]["accumulation_steps"],grad_clip_norm=c["optimizer"]["grad_clip_norm"])
        v=evaluate_loader(view,vl,criterion,device);tm=summarize(t,policy);vm=summarize(v,policy)
        improved,stop=early.update(vm["main_mAP"],epoch)
        row=dict(epoch=epoch,lr=optimizer.param_groups[0]["lr"],train_online_BCE118=tm["BCE_Train_supported118"],valid_BCE118=vm["BCE_Train_supported118"],
            train_online_objective=t["objective"],valid_objective=v["objective"],train_online_BCE119=tm["BCE119"],valid_BCE119=vm["BCE119"],train_online_Macro_F1_at_05=tm["main_macro_f1"],valid_Macro_F1_at_05=vm["main_macro_f1"],train_online_mAP=tm["main_mAP"],valid_mAP=vm["main_mAP"],seconds=time.perf_counter()-began)
        history.append(row)
        state=dict(format="animalact2-research-tools-v1",family=a.family,config=c,epoch=epoch,head_state={k:p.detach().cpu() for k,p in model.state_dict().items() if not k.startswith("backbone.")},
            optimizer=optimizer.state_dict(),class_policy=policy.state_dict(),schedule=schedule.state_dict(),early=early.state_dict(),RNG=rng_state(),history=history,
            data_fingerprints=development_fingerprints(),cache_fingerprints=cache_fp,implementation_fingerprints=source_fp)
        torch.save(state,out/"last.pending");(out/"last.pending").replace(out/"last.pt")
        if improved:torch.save(state,out/"best.pt")
        (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        with (out/"history.csv").open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        emit(row)
        if stop:break
    load_head(out/"best.pt");v=evaluate_loader(view,vl,criterion,device)
    threshold=select_validation_threshold(v["logits"],v["target"],split="valid");metrics=summarize(v,policy,threshold=threshold["threshold"])
    (out/"threshold_selection.json").write_text(json.dumps(threshold,indent=2),encoding="utf-8")
    (out/"results.json").write_text(json.dumps(dict(valid=metrics,best_epoch=early.best_epoch,Test_read=False),indent=2),encoding="utf-8")
    emit(dict(status="completed",output=str(out),best_epoch=early.best_epoch,valid_mAP=metrics["main_mAP"]))
if __name__=="__main__":main()
