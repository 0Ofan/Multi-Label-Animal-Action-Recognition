"""Final saved-result audit and plots for cached VideoMAE heads; no Test access."""
import csv
import gc
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import torch
from animalact2.phase1_5.config import config_hash,cohort_hash,development_fingerprints
from .models import load_config,build_model,KINDS
from animalact2.phase1_5.losses import UnsupportedClassPolicy
from animalact2.phase1_5.schedule import LearningRatePlan,EarlyStopping
from animalact2.phase1_5.evaluation import evaluate_outputs,select_validation_threshold
from .data import CachedFeatureDataset,collate_features
from .support import ROOT,OUT,enable_guard,write_json,sha
from . import checkpoint


def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plot_curves():
    # All Matplotlib runtime caches remain inside the new experiment directory.
    os.environ["MPLCONFIGDIR"]=str(OUT/".runtime/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in ("model1","model2","model3"):
        d=OUT/"runs"/f"{name}_bce_seed42"
        history=json.loads((d/"history.json").read_text(encoding="utf-8"))
        best=max(history,key=lambda r:r["valid_mAP"])["epoch"]
        specs=[("BCE objective: 118 Train-supported classes","train_online_objective","valid_objective",False),
               ("BCE119: diagnostic including unsupported class","train_online_BCE119","valid_BCE119",False),
               ("Fixed31 mAP","train_online_mAP","valid_mAP",True)]
        fig,axes=plt.subplots(3,1,figsize=(10,10),layout="constrained")
        for ax,(title,t,v,percent) in zip(axes,specs):
            factor=100 if percent else 1
            x=[r["epoch"] for r in history]
            ax.plot(x,[r[t]*factor for r in history],color="#2563eb",label="Train online (mixed parameter states)")
            ax.plot(x,[r[v]*factor for r in history],color="#dc2626",label="Valid end-of-epoch checkpoint")
            ax.axvline(best,color="#64748b",ls="--",lw=1,label=f"Best Valid mAP: epoch {best}")
            ax.set(title=title,xlabel="Epoch",ylabel="mAP (%)" if percent else "BCE",xlim=(1,len(history)))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f"{name.upper()} | VideoMAE | release-v1 | seed 42 | BCE")
        fig.savefig(d/"training_curves.svg")
        fig.savefig(d/"training_curves.png",dpi=140)
        plt.close(fig)
    write_json(OUT/"environment.json",dict(python=platform.python_version(),interpreter=sys.executable,
                torch=torch.__version__,CUDA_runtime=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
                GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,matplotlib=matplotlib.__version__,
                torch_CPU_threads=4,precision="fp32",network_or_dependency_install_used=False))

def audit():
    started=time.perf_counter();torch.set_num_threads(4)
    status=json.loads((OUT/"run_status.json").read_text(encoding="utf-8"))
    assert status["all_complete"] and not status["Test_used"] and not status["extra_experiments_started"]
    assert set(status["models"])=={"model1","model2","model3"}
    gate=json.loads((OUT/"integration_validation.json").read_text(encoding="utf-8"))
    assert gate["status"]=="passed" and gate["integration_fingerprints"]==checkpoint.fingerprints()
    train,valid=CachedFeatureDataset("train"),CachedFeatureDataset("valid")
    assert (len(train),len(valid))==(4667,979)
    counts=torch.tensor([r["target_119"] for r in train.records]).sum(0)
    expected_y=torch.tensor([r["target_119"] for r in valid.records],dtype=torch.float32)
    expected_ids=[r["clip_id"] for r in valid.records]
    expected_parents=[r["parent_video_id"] for r in valid.records]
    first_batch=collate_features([valid[i] for i in range(8)])
    mixed_indices=[next(i for i,r in enumerate(valid.records) if r["effective_length"]==n) for n in (1,3,4,16)]
    mixed=collate_features([valid[i] for i in mixed_indices])
    assert mixed["lengths"].tolist()==[1,2,2,8]
    assert torch.equal(mixed["padding_mask"],~mixed["valid_frame_mask"])
    comparisons=[];details={};orders={};targets={};all_metrics={};configs={}
    for name in ("model1","model2","model3"):
        directory=OUT/"runs"/f"{name}_bce_seed42"
        cfg=load_config(OUT/f"configs/{name}_seed42_bce.json")
        result=json.loads((directory/"results.json").read_text(encoding="utf-8"))
        hist=json.loads((directory/"history.json").read_text(encoding="utf-8"))
        with (directory/"history.csv").open(encoding="utf-8-sig",newline="") as f:csvhist=list(csv.DictReader(f))
        assert len(csvhist)==len(hist)
        assert result["status"]=="completed" and result["config"]==cfg
        assert json.loads((directory/"config.json").read_text(encoding="utf-8"))==cfg
        assert result["config_SHA256"]==config_hash(cfg) and result["shared_cohort_SHA256"]==cohort_hash(cfg)
        assert result["cache_provenance"]==checkpoint.cache_provenance()
        assert not result["Test_used"] and not result["ASL_or_additional_experiments"] and result["encoder_forward_calls"]==0
        assert 1<=len(hist)<=50 and result["completed_epochs"]==len(hist)
        dummy=torch.nn.Parameter(torch.zeros(()));optim=torch.optim.AdamW([dummy],lr=.001)
        plan=LearningRatePlan(optim,cfg);early=EarlyStopping(10,0.)
        for epoch,row in enumerate(hist,1):
            assert row["epoch"]==epoch and row["optimizer_updates"]==146 and row["encoder_forward_calls"]==0
            assert abs(row["lr"]-plan.apply_epoch(epoch))<1e-15
            _,stop=early.update(row["valid_mAP"],epoch)
            assert row["bad_epochs"]==early.bad_epochs and row["best_epoch"]==early.best_epoch
            if epoch<len(hist):assert not stop
            for key in ("valid_mAP","train_online_mAP","valid_Macro_F1_at_05","train_online_Macro_F1_at_05"):
                assert math.isfinite(row[key]) and 0<=row[key]<=1
            for key in ("valid_objective","train_online_objective","epoch_seconds"):
                assert math.isfinite(row[key]) and row[key]>0
            orders.setdefault(epoch,set()).add(row["Train_input_order_SHA256"])
            targets.setdefault(epoch,set()).add(row["Train_targets_SHA256"])
        assert len(hist)==50 or early.bad_epochs>=10
        expected_best=max(hist,key=lambda r:r["valid_mAP"])
        assert result["best_epoch"]==expected_best["epoch"]==early.best_epoch
        model=build_model(cfg).to("cuda").eval()
        policy=UnsupportedClassPolicy(model,counts)
        best=checkpoint.load(directory/"best.pt",model,cfg,policy=policy)
        last=torch.load(directory/"last.pt",map_location="cpu",weights_only=True)
        assert not best["diagnostic"] and not last["diagnostic"]
        assert best["epoch"]==expected_best["epoch"] and last["epoch"]==len(hist)
        assert last["history"]==hist and best["history"]==hist[:best["epoch"]]
        assert last["early_stopping"]==early.state_dict()
        assert last["data_fingerprints"]==development_fingerprints()
        assert last["first_round_implementation"]==checkpoint.fingerprints()
        for saved in (best,last):
            assert saved["cache_provenance"]==checkpoint.cache_provenance()
            assert saved["config"]==cfg and saved["config_SHA256"]==config_hash(cfg)
            assert torch.equal(saved["class_policy"]["train_positive_counts"],counts)
            assert saved["class_policy"]["unsupported_ids"].tolist()==[23]
            assert saved["optimizer"]["state"] and saved["RNG"]["torch"].numel()>0
            assert all(torch.isfinite(t).all() for t in saved["head_state"].values())
            assert all(torch.equal(saved["head_state"][k][[23]],v) for k,v in saved["class_policy"]["anchors"].items())
            assert not any(k.startswith("backbone.") for k in saved["head_state"])
        pred=torch.load(directory/"valid_predictions.pt",map_location="cpu",weights_only=True)
        assert pred["logits"].shape==(979,119) and torch.isfinite(pred["logits"]).all()
        assert torch.equal(pred["target"],expected_y)
        assert pred["clip_ids"]==expected_ids and pred["parent_video_ids"]==expected_parents
        assert len(set(pred["clip_ids"]))==979
        assert pred["checkpoint_SHA256"]==result["checkpoint_SHA256"]==sha(directory/"best.pt")
        assert pred["cache_provenance"]==checkpoint.cache_provenance()
        threshold=select_validation_threshold(pred["logits"],pred["target"],split="valid")
        assert threshold==result["threshold_selection"]==json.loads((directory/"threshold_selection.json").read_text(encoding="utf-8"))
        metrics=evaluate_outputs(pred["logits"],pred["target"],pred["parent_video_ids"],policy,threshold=threshold["threshold"])
        assert abs(result["valid"]["training_objective"]-expected_best["valid_objective"])<1e-8
        assert abs(result["valid"]["training_objective"]-metrics["BCE_Train_supported118"])<1e-7
        metrics["training_objective"]=result["valid"]["training_objective"]
        assert metrics==result["valid"]==json.loads((directory/"best_valid/metrics.json").read_text(encoding="utf-8"))
        assert metrics["main_mAP"]==expected_best["valid_mAP"]
        assert len(metrics["main_class_ids"])==31 and 23 not in metrics["main_class_ids"]
        for file,count in (("per_class_119.csv",119),("per_class_main31.csv",31),("confusion_main31.csv",31)):
            with (directory/"best_valid"/file).open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
            assert len(rows)==count
            for row in rows:
                r=metrics["per_class"][int(row["class_id"])]
                assert row["name"]==r["name"]
                for key in ("TP","FP","TN","FN"):assert int(row[key])==r[key]
                assert sum(r[k] for k in ("TP","FP","TN","FN"))==979
        encoder_calls=[]
        def no_encoder(module,args):
            encoder_calls.append(True);raise AssertionError("Cached checkpoint audit cannot invoke the encoder")
        hook=model.backbone.encoder.register_forward_pre_hook(no_encoder)
        with torch.inference_mode():
            z=model.forward_features(first_batch["frames"].cuda(),first_batch["lengths"],first_batch["valid_frame_mask"].cuda())
            mf=mixed["frames"].cuda();mm=mixed["valid_frame_mask"].cuda()
            original=model.forward_features(mf,mixed["lengths"],mm,return_aux=True)
            poisoned=mf.clone();poisoned[~mm]=12345.
            changed=model.forward_features(poisoned,mixed["lengths"],mm)
            padding_error=float((original["logits"]-changed).abs().max())
            assert original["logits"].shape==(4,119) and torch.isfinite(original["logits"]).all()
            assert padding_error==0.
            if name=="model3":
                attention=original["temporal_attention"]
                assert attention.shape==(4,119,8) and torch.all(attention.masked_select(~mm[:,None,:])==0)
                torch.testing.assert_close(attention.sum(-1),torch.ones(4,119,device="cuda"))
        error=float((z.cpu()-pred["logits"][:8]).abs().max())
        assert error<=1e-5 and not encoder_calls
        hook.remove()
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        total=sum(p.numel() for p in model.parameters())
        resource=result["resources"]
        comparisons.append(dict(model=name,seed=42,loss="BCE_Train_supported118",Train_clips=4667,Valid_clips=979,
             best_epoch=best["epoch"],completed_epochs=len(hist),stop_reason=result["stop_reason"],
             valid_mAP31=metrics["main_mAP"],valid_Macro_F1_31=metrics["main_macro_f1"],valid_BCE119=metrics["BCE119"],
             valid_BCE118=metrics["BCE_Train_supported118"],valid_selected_threshold=threshold["threshold"],
             train_online_mAP_at_best=hist[best["epoch"]-1]["train_online_mAP"],
             best_valid_objective=hist[best["epoch"]-1]["valid_objective"],last_train_online_objective=hist[-1]["train_online_objective"],
             min_valid_objective=min(r["valid_objective"] for r in hist),valid_loss_min_epoch=min(hist,key=lambda r:r["valid_objective"])["epoch"],
             last_valid_objective=hist[-1]["valid_objective"],last_valid_mAP31=hist[-1]["valid_mAP"],
             wall_seconds=resource["wall_seconds"],epoch_core_seconds=result["training_epoch_seconds"],
             median_epoch_seconds=sorted(r["epoch_seconds"] for r in hist)[len(hist)//2],
             CPU_RSS_parent_peak_MiB=resource["CPU_RSS_parent_peak_bytes"]/2**20,
             CPU_RSS_parent_workers_sum_peak_MiB=resource["CPU_RSS_parent_workers_sum_peak_bytes"]/2**20,
             GPU_allocated_peak_MiB=resource["GPU_allocated_peak_bytes"]/2**20,
             GPU_reserved_peak_MiB=resource["GPU_reserved_peak_bytes"]/2**20,
             total_parameters=total,trainable_parameter_tensor_elements=trainable,resume_count=resource["resume_count"],
             config_SHA256=config_hash(cfg),shared_cohort_SHA256=cohort_hash(cfg)))
        details[name]=dict(status="passed",checkpoint_prefix_history_valid=True,strict_best_valid_epoch=best["epoch"],
                          restored_first8_Valid_max_logit_error=error,Valid_rows=979,main_classes=31,
                          mixed_real_lengths=[1,3,4,16],padded_feature_perturbation_max_logit_error=padding_error,
                          unsupported_rows_unchanged=True,threshold_source="Valid only",encoder_calls=0,
                          checkpoint_SHA256=result["checkpoint_SHA256"],all_119_confusion_counts_aligned=True)
        all_metrics[name]=metrics;configs[name]=cfg
        del model,policy,best,last,pred,z,mf,mm,original,poisoned,changed;gc.collect();torch.cuda.empty_cache()
    assert len({cohort_hash(c) for c in configs.values()})==1
    assert all(len(s)==1 for s in orders.values()) and all(len(s)==1 for s in targets.values())
    assert all(all_metrics[n]["main_class_ids"]==all_metrics["model1"]["main_class_ids"] for n in all_metrics)
    per_class=[]
    for i in all_metrics["model1"]["main_class_ids"]:
        base=all_metrics["model1"]["per_class"][i]
        row=dict(class_id=i,name=base["name"],Train_positive_clips=int(counts[i]),Valid_positive_clips=base["positive_clips"],
                 Valid_positive_parent_videos=base["positive_parent_videos"])
        for n,m in all_metrics.items():
            r=m["per_class"][i];assert r["name"]==base["name"] and r["positive_clips"]==base["positive_clips"]
            row.update({n+"_"+k:r[k] for k in ("AP","Precision","Recall","F1")})
        row["model2_minus_model1_AP"]=row["model2_AP"]-row["model1_AP"]
        row["model3_minus_model1_AP"]=row["model3_AP"]-row["model1_AP"]
        row["model3_minus_model2_AP"]=row["model3_AP"]-row["model2_AP"]
        per_class.append(row)
    write_csv(OUT/"comparison.csv",comparisons)
    write_csv(OUT/"per_class_comparison_main31.csv",per_class)
    write_json(OUT/"comparison.json",comparisons)
    result=dict(status="passed",models=details,shared_protocol_equal=True,shared_Train_epoch_order_and_targets=True,
                unique_Train=4667,unique_Valid=979,updates_per_epoch=146,Test_read=False,seconds=time.perf_counter()-started)
    write_json(OUT/"experiment_audit.json",result)
    return comparisons,per_class,result


def compare_backbones(rows,per_class):
    """Existing ResNet Valid summaries only; no old checkpoints/predictions are loaded."""
    combined=[];paired=[];old_results={};class_rows=[]
    for r in rows:
        name=r["model"];d=ROOT/f"animalact2/first_round/runs/{name}_bce_seed42"
        old=json.loads((d/"results.json").read_text(encoding="utf-8"));old_results[name]=old
        new=json.loads((OUT/f"runs/{name}_bce_seed42/results.json").read_text(encoding="utf-8"))
        assert old["valid"]["num_samples"]==new["valid"]["num_samples"]==979
        assert old["valid"]["main_class_ids"]==new["valid"]["main_class_ids"]
        assert {k:v for k,v in old["config"].items() if k!="backbone"}=={k:v for k,v in new["config"].items() if k not in ("backbone","input_cache")}
        oh=json.loads((d/"history.json").read_text(encoding="utf-8"))
        nh=json.loads((OUT/f"runs/{name}_bce_seed42/history.json").read_text(encoding="utf-8"))
        for a,b in zip(oh,nh):
            assert a["Train_input_order_SHA256"]==b["Train_input_order_SHA256"] and a["Train_targets_SHA256"]==b["Train_targets_SHA256"]
        for backbone,record,params in (("ResNet18",old,{"model1":61047,"model2":523639,"model3":554103}[name]),("VideoMAE",new,r["trainable_parameter_tensor_elements"])):
            m=record["valid"]
            combined.append(dict(backbone=backbone,model=name,head=KINDS[name],seed=42,best_epoch=record["best_epoch"],epochs=record["completed_epochs"],
                                 valid_mAP31=m["main_mAP"],valid_Macro_F1_31=m["main_macro_f1"],valid_BCE119=m["BCE119"],
                                 valid_BCE118=m["BCE_Train_supported118"],Valid_selected_threshold=m["threshold"],trainable_head_parameters=params))
        per=[]
        for i in new["valid"]["main_class_ids"]:
            a=old["valid"]["per_class"][i];b=new["valid"]["per_class"][i]
            assert a["name"]==b["name"] and a["positive_clips"]==b["positive_clips"]
            per.append(b["AP"]-a["AP"])
        paired.append(dict(model=name,head=KINDS[name],mAP_delta=new["valid"]["main_mAP"]-old["valid"]["main_mAP"],
                           Macro_F1_delta=new["valid"]["main_macro_f1"]-old["valid"]["main_macro_f1"],
                           BCE119_delta=new["valid"]["BCE119"]-old["valid"]["BCE119"],BCE118_delta=new["valid"]["BCE_Train_supported118"]-old["valid"]["BCE_Train_supported118"],
                           classes_AP_up=sum(x>1e-12 for x in per),classes_AP_down=sum(x<-1e-12 for x in per),classes_AP_tied=sum(abs(x)<=1e-12 for x in per)))
    for row in per_class:
        result={k:row[k] for k in ("class_id","name","Train_positive_clips","Valid_positive_clips")};i=row["class_id"]
        for name in KINDS:
            result[name+"_ResNet_AP"]=old_results[name]["valid"]["per_class"][i]["AP"]
            result[name+"_VideoMAE_AP"]=row[name+"_AP"]
            result[name+"_delta_AP"]=row[name+"_AP"]-result[name+"_ResNet_AP"]
        class_rows.append(result)
    write_csv(OUT/"backbone_comparison.csv",combined);write_csv(OUT/"backbone_paired_changes.csv",paired)
    write_csv(OUT/"backbone_per_class_main31.csv",class_rows)
    write_json(OUT/"backbone_comparison_checks.json",dict(status="passed",protocol_fields_equal=True,matching_epoch_order_and_labels=True,
                                                        class_mapping_and_Valid_support_equal=True,Test_read=False))
    return combined,paired


def report(rows,per_class,paired,protection):
    pct=lambda x:f"{100*x:.2f}%"
    best=max(rows,key=lambda r:r["valid_mAP31"])
    best_f1=max(rows,key=lambda r:r["valid_Macro_F1_31"])
    table="\n".join(f"| {KINDS[r['model']]} | {r['completed_epochs']} | {r['best_epoch']} | {pct(r['valid_mAP31'])} | {pct(r['valid_Macro_F1_31'])} | {r['valid_BCE119']:.6f} | {r['valid_BCE118']:.6f} | {r['valid_selected_threshold']:.2f} | {r['wall_seconds']/60:.2f} |" for r in rows)
    deltas="\n".join(f"| {p['head']} | {100*p['mAP_delta']:+.2f} | {100*p['Macro_F1_delta']:+.2f} | {p['BCE118_delta']:+.6f} | {p['classes_AP_up']} / {p['classes_AP_down']} / {p['classes_AP_tied']} |" for p in paired)
    resources="\n".join(f"| {KINDS[r['model']]} | {r['trainable_parameter_tensor_elements']:,} | {r['GPU_allocated_peak_MiB']:.1f} | {r['GPU_reserved_peak_MiB']:.1f} | {r['CPU_RSS_parent_peak_MiB']:.1f} | {r['CPU_RSS_parent_workers_sum_peak_MiB']:.1f} | {r['median_epoch_seconds']:.2f} |" for r in rows)
    curves="\n".join(f"- {KINDS[r['model']]}：最佳轮Train online mAP {pct(r['train_online_mAP_at_best'])}，Valid {pct(r['valid_mAP31'])}；Valid BCE118最低点为第{r['valid_loss_min_epoch']}轮（{r['min_valid_objective']:.6f}），最终为{r['last_valid_objective']:.6f}；最终Train online BCE118为{r['last_train_online_objective']:.6f}。停止原因：{r['stop_reason']}。" for r in rows)
    runner=json.loads((OUT/"run_status.json").read_text(encoding="utf-8"))
    text=f"""# AnimalAct 2.0 Phase 2.4 VideoMAE第一轮正式训练报告

**三个分类头均已完成正式训练、最佳Checkpoint恢复、Valid结果导出及最终核验。停止等待审核；没有Test评价或下一轮实验。**

## 数据、结构与冻结范围

使用 `animalact2-2.0.0-release-v1` 的固定Train4667、Valid979。读取已封存的Phase2.3 VideoMAE FP32变长缓存 `[T,768]`，T=ceil(原真实帧长度/2)=1–8；保留原119维标签、父视频、Clip、真实帧及标注来源。输入是已经建模全局时空关系的tubelet特征，不是独立逐帧CNN特征。

直接复用Phase2.2已验证的三个分类头：

1. Mean：有效时间Mean Pooling → Linear(768,119)。
2. BiGRU Mean：单层双向GRU，hidden=128/方向 → 有效时间Mean Pooling → Linear(256,119)。
3. BiGRU Attention：同一BiGRU → 119类对应的Masked Temporal Attention → 类别对应分类权重 → 119 Logits。

没有额外投影、Spatial Attention、Dropout或新模块。变长序列只在临时Batch Padding，BiGRU使用真实长度打包，Mean/Attention屏蔽无效步；部分真实的末tubelet仍有效，完全沿用Pilot。

视觉Backbone为缓存来源对应的冻结Kinetics预训练VideoMAE。**本训练进程完全不构造、加载或运行86,225,664参数的视觉编码器**；只记录一个无参数的冻结来源引用，用于Checkpoint来源检查和调用阻断。没有原视频解码、源视频哈希重读、特征重提取或Backbone微调。模型注册的参数总数等于任务头参数，不能把它误认为整个VideoMAE系统的参数量。

## 与ResNet18一致的协议

Seed42、BCE、Batch8、累积4（有效Batch通常32）、AdamW、峰值LR1e-3、WD1e-4、梯度裁剪5、前2轮Warmup、Cosine到峰值10%、最多50轮、Patience10、FP32；每轮完整遍历Train一次，共146次更新。最后不足32样本的累积窗口按真实样本数归一化。DataLoader沿用2 workers、prefetch1、pin_memory及固定每轮洗牌规则。

复用既有 `phase1_5.engine` 的Train/Valid Epoch、损失与评价函数、优化器、调度、早停和随机状态机制。配置除Backbone来源和输入缓存说明外，与ResNet18对应模型逐字段一致；同一Epoch的Clip顺序及目标SHA也完全一致。

三个头各自重新Seed42初始化，不继承诊断权重、其它头或AnimalAct1.0权重。严格按Valid固定31类mAP选最优轮，平分保留较早轮。最佳模型在完整Valid上用0.05–0.95、步长0.01选择单一Macro-F1最优阈值，平分取较低值；不使用Test。

Defensive Rearing（ID23）Train无正例：保留119维Logits，排除该类训练损失，类别专属参数行在每次AdamW后恢复以抵消Weight Decay，标记为不可正常发出动作；不修改标签定义。实际优化为**BCE118**。**BCE119**保留该不可学习输出作为原始诊断值；固定31主类不含该类。

## 最佳Valid结果

| 分类头 | 完成轮数 | 最佳轮 | mAP31 | Macro-F1 | BCE119 | BCE118 | Valid阈值 | 总耗时/分钟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

本轮按预设主指标最优为 **{KINDS[best['model']]}，Valid mAP {pct(best['valid_mAP31'])}**。这是单随机种子、同一Valid选模结果，不是Test泛化结论或统计显著性结论。

按Macro-F1排序，最高为 **{KINDS[best_f1['model']]}：{pct(best_f1['valid_Macro_F1_31'])}**。mAP与F1排序不同，继续沿用原定mAP主指标选模，完整保留三个候选及各自Valid阈值。

三头顺序运行总耗时约 **{runner['runner_seconds']/60:.2f}分钟**，包括初始化和清理；各模型内部训练/Valid/保存/最终导出累计 **{sum(r['wall_seconds'] for r in rows)/60:.2f}分钟**。整合开发、前置诊断、最终核验及报告不计入训练耗时。

## 与已有ResNet18第一轮比较

下表为VideoMAE减去相同头结构的ResNet18，百分点用于mAP/F1，BCE为绝对差值：

| 分类头 | mAP变化/百分点 | Macro-F1变化/百分点 | BCE118变化 | 31类AP提升/下降/相同 |
|---|---:|---:|---:|---|
{deltas}

完整六模型表见 `backbone_comparison.csv`，逐类对照见 `backbone_per_class_main31.csv`。标签映射、Valid正例支持数及评价规则逐项核验一致，只读取已有ResNet18 Valid汇总和历史，不读取旧预测或Test。

比较属于**冻结预训练Backbone方案比较**：ImageNet与Kinetics预训练不同，特征512→768、时间步1–16→1–8，分类头输入参数量随之变化。不能将差值归因于单独某一网络结构，也不能直接与AnimalAct1.0视频级mAP比较。

## 曲线、过拟合与资源

{curves}

Train Loss下降、Valid Loss从最低点回升时，提示概率泛化存在过拟合。BCE最小轮与mAP最优轮不必一致：前者衡量概率拟合，后者衡量类别内排序。本轮不因观察结果而改变既定选模规则。

Train mAP/F1/Loss为Epoch内更新前的online混合参数状态，Valid是Epoch结束的固定模型，两者不是同一固定Checkpoint的精确泛化差距。历史F1阈值0.5；上表最终F1使用最佳模型的Valid所选阈值。`training_curves.png/svg`展示BCE118、BCE119和mAP31，完整数值见history CSV/JSON/JSONL。

| 分类头 | 任务头参数 | GPU分配峰值/MiB | GPU保留峰值/MiB | 主进程RSS/MiB | 主进程+Workers RSS/MiB | Epoch中位耗时/秒 |
|---|---:|---:|---:|---:|---:|---:|
{resources}

CUDA设备为RTX5060 Laptop GPU。显存为PyTorch分配/保留峰值，不含驱动及桌面。RAM为Windows Working Set，20ms采样；主进程+Workers之和可能重复计算共享页，不等于独占物理RAM。每轮耗时含Train、Valid和指标，不含Checkpoint/日志I/O；总模型耗时包含它们及最终Valid导出。

## 整合与完成核验

- 训练前校验Full Cache **11,323个交付文件SHA256**，包含全部5646份数组/收据及范围/来源说明。
- 每个头先对11个真实Train Clip完成一次诊断更新，完整979个Valid上运行评价和阈值搜索；Loss有限、各参数张量更新、ID23参数行固定、Padding扰动不改变Logits。诊断分数不是正式结果，也未用于正式选模。
- Checkpoint恢复后，完整Valid Logits逐位一致；AdamW、调度、早停以及Python/NumPy/Torch CPU/CUDA随机状态恢复。诊断Checkpoint拒绝初始化正式运行。
- 最终核验Best/Last状态、完整历史前缀、严格最佳轮、早停、逐轮146次更新、学习率、Clip顺序、119维真实标签、31类AP与2×2混淆统计。保存的Valid Logits重算mAP/F1/BCE和阈值，与results.json完全一致。
- 最佳Checkpoint另用8条Valid缓存复核恢复Logits，并检查真实长度1/3/4/16对应特征长度1/2/2/8的Padding不变性和Attention Mask。结果见 `experiment_audit.json`，不调用编码器。
- 历史保护：{protection['historical_metadata_files']:,}个旧文件元数据、{protection['historical_hashed_files']:,}个受保护文件内容SHA复核不变，包含release、Full Cache、Pilot和ResNet18第一轮。旧Checkpoint仅在保护工具中按字节哈希，不作为本轮模型载入。

## 交付与停止

- `runs/modelN_bce_seed42/`：Best/Last Checkpoint、完整配置、来源、全部历史、曲线、results.json、Valid原始Logits/target/Clip ID、阈值搜索，以及119类/31类AP和混淆表。
- `comparison.csv/json`：三个VideoMAE分类头统一结果、资源、阈值、配置哈希。
- `per_class_comparison_main31.csv`：三头逐类AP、Precision/Recall/F1及配对变化。
- `backbone_comparison.csv`、`backbone_paired_changes.csv`、`backbone_per_class_main31.csv`：现有ResNet18与新VideoMAE的Valid对照。
- `preflight/`、`integration_validation.json`、`experiment_audit.json`、`backbone_comparison_checks.json`、`protection_verification.json`、`artifact_manifest.json`、`FINAL_VERIFICATION.json`：核验依据和SHA清单。

Checkpoint包含全部可学习头权重、AdamW、调度、早停、RNG、历史及完整缓存/数据/实现指纹；冻结编码器只保存来源引用。`run_all --execute-training` 对未完成实验使用last.pt恢复，已完成则跳过，禁止覆盖旧实验。

全部新代码和结果仅在 `animalact2/phase2_4/`。没有修改ResNet18/Pilot/Full Cache，没有读取Test视频、标注、Manifest或预测，没有ASL、微调、额外消融、特征提取、模型下载或前端开发。

**三个分类头全部完成后停止，等待人工审核，不自动进行Test评价或下一轮实验。**
"""
    (OUT/"FINAL_PHASE2_4_REPORT.md").write_text(text,encoding="utf-8")


def main():
    enable_guard()
    if (OUT/"FINAL_VERIFICATION.json").exists():raise FileExistsError("Preserve completed delivery")
    rows,per_class,checks=audit();plot_curves();_,paired=compare_backbones(rows,per_class)
    protection=json.loads((OUT/"protection_verification.json").read_text(encoding="utf-8"));assert protection["status"]=="passed"
    report(rows,per_class,paired,protection)
    artifacts={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob("*"))
               if p.is_file() and ".runtime" not in p.parts and p.name not in ("FINAL_VERIFICATION.json","artifact_manifest.json")}
    write_json(OUT/"artifact_manifest.json",dict(files=artifacts,excludes=["artifact_manifest.json","FINAL_VERIFICATION.json",".runtime/"]))
    write_json(OUT/"FINAL_VERIFICATION.json",dict(status="passed",all_models_complete=True,experiment_audit=checks,protection=protection,
                files_in_manifest=len(artifacts),logical_bytes=sum(v["bytes"] for v in artifacts.values()),
                artifact_manifest_SHA256=sha(OUT/"artifact_manifest.json"),Test_read=False,second_round_started=False))
    print(json.dumps(dict(status="passed",report=str(OUT/"FINAL_PHASE2_4_REPORT.md"),models=rows)),flush=True)


if __name__=="__main__":main()
