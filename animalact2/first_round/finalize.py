"""Audit saved first-round artifacts and export comparison/report; Train/Valid only."""
from __future__ import annotations
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
from animalact2.phase1_5.config import load_config,config_hash,cohort_hash,development_fingerprints
from animalact2.phase1_5.models import build_model
from animalact2.phase1_5.losses import UnsupportedClassPolicy
from animalact2.phase1_5.schedule import LearningRatePlan,EarlyStopping
from animalact2.phase1_5.evaluation import evaluate_outputs,select_validation_threshold
from animalact2.phase1_5_cache_full.dataset import CachedFeatureDataset,collate_features
from .support import ROOT,OUT,enable_guard,write_json,sha,verify
from . import checkpoint


def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot_curves():
    # All Matplotlib runtime caches remain inside the new experiment directory.
    os.environ["MPLCONFIGDIR"]=str(OUT/".runtime_temp/matplotlib")
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
        fig.suptitle(f"{name.upper()} | AnimalAct2 release-v1 | seed 42 | BCE")
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
    assert mixed["lengths"].tolist()==[1,3,4,16]
    assert torch.equal(mixed["padding_mask"],~mixed["valid_frame_mask"])
    assert torch.all(mixed["frame_indices"][mixed["padding_mask"]]==-1)
    comparisons=[];details={};orders={};targets={};all_metrics={};configs={}
    for name in ("model1","model2","model3"):
        directory=OUT/"runs"/f"{name}_bce_seed42"
        cfg=load_config(ROOT/f"animalact2/phase1_5/configs/{name}_seed42_bce.json")
        result=json.loads((directory/"results.json").read_text(encoding="utf-8"))
        hist=json.loads((directory/"history.json").read_text(encoding="utf-8"))
        with (directory/"history.csv").open(encoding="utf-8-sig",newline="") as f:csvhist=list(csv.DictReader(f))
        assert len(csvhist)==len(hist)
        assert result["status"]=="completed" and result["config"]==cfg
        assert json.loads((directory/"config.json").read_text(encoding="utf-8"))==cfg
        assert result["config_SHA256"]==config_hash(cfg) and result["shared_cohort_SHA256"]==cohort_hash(cfg)
        assert result["cache_provenance"]==checkpoint.cache_provenance()
        assert not result["Test_used"] and not result["ASL_or_additional_experiments"] and result["CNN_forward_calls"]==0
        assert 1<=len(hist)<=50 and result["completed_epochs"]==len(hist)
        dummy=torch.nn.Parameter(torch.zeros(()));optim=torch.optim.AdamW([dummy],lr=.001)
        plan=LearningRatePlan(optim,cfg);early=EarlyStopping(10,0.)
        for epoch,row in enumerate(hist,1):
            assert row["epoch"]==epoch and row["optimizer_updates"]==146 and row["CNN_forward_calls"]==0
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
        cnn_calls=[]
        def no_CNN(module,args):
            cnn_calls.append(True);raise AssertionError("Cached checkpoint audit cannot invoke CNN")
        hook=model.backbone.encoder.register_forward_pre_hook(no_CNN)
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
                assert attention.shape==(4,119,16) and torch.all(attention.masked_select(~mm[:,None,:])==0)
                torch.testing.assert_close(attention.sum(-1),torch.ones(4,119,device="cuda"))
        error=float((z.cpu()-pred["logits"][:8]).abs().max())
        assert error<=1e-5 and not cnn_calls
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
                          unsupported_rows_unchanged=True,threshold_source="Valid only",CNN_calls=0,
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


def report(rows,per_class,protection):
    names={"model1":"ResNet18 + Mean Pooling","model2":"ResNet18 + BiGRU + Mean Pooling","model3":"ResNet18 + BiGRU + Class-specific Temporal Attention"}
    p=lambda x:f"{100*x:.2f}%"
    table="\n".join(f"| {r['model']} | {r['completed_epochs']} | {r['best_epoch']} | {p(r['valid_mAP31'])} | {p(r['valid_Macro_F1_31'])} | {r['valid_BCE119']:.6f} | {r['valid_BCE118']:.6f} | {r['valid_selected_threshold']:.2f} | {r['wall_seconds']/60:.2f} |" for r in rows)
    resource="\n".join(f"| {r['model']} | {r['trainable_parameter_tensor_elements']:,} | {r['GPU_allocated_peak_MiB']:.1f} | {r['GPU_reserved_peak_MiB']:.1f} | {r['CPU_RSS_parent_peak_MiB']:.1f} | {r['CPU_RSS_parent_workers_sum_peak_MiB']:.1f} | {r['median_epoch_seconds']:.2f} |" for r in rows)
    best=max(rows,key=lambda r:r["valid_mAP31"])
    runner_seconds=json.loads((OUT/"run_status.json").read_text(encoding="utf-8"))["runner_seconds"]
    pair=[]
    for a,b in ((rows[1],rows[0]),(rows[2],rows[0]),(rows[2],rows[1])):
        key=a["model"]+"_minus_"+b["model"]+"_AP"
        up=sum(r[key]>1e-12 for r in per_class);down=sum(r[key]<-1e-12 for r in per_class)
        pair.append(f"- {a['model']} 相对 {b['model']}：Valid mAP {100*(a['valid_mAP31']-b['valid_mAP31']):+.2f} 个百分点，Macro-F1 {100*(a['valid_Macro_F1_31']-b['valid_Macro_F1_31']):+.2f} 个百分点；逐类 AP 提升 {up} 类、下降 {down} 类、相同 {31-up-down} 类。")
    curves="\n".join(f"- `{r['model']}`：最佳轮 Train online mAP {p(r['train_online_mAP_at_best'])}，Valid mAP {p(r['valid_mAP31'])}；Valid BCE118 最低点在第 {r['valid_loss_min_epoch']} 轮（{r['min_valid_objective']:.6f}），最终为 {r['last_valid_objective']:.6f}；最终 Train online BCE118 {r['last_train_online_objective']:.6f}；停止原因 `{r['stop_reason']}`。" for r in rows)
    text=f"""# AnimalAct 2.0 第一轮正式实验报告

## 完成范围

已完成 Full Cache 入口整合、三个模型的小样本集成验证、三次独立正式训练、最佳模型恢复及全部交付结果核验。所有新增内容均位于 `animalact2/first_round/`。没有读取 Test 视频、标注或预测，没有提取新特征，没有启动 ASL 或其他消融，也没有进入下一轮实验。

正式数据版本 `animalact2-2.0.0-release-v1`，Train 4667、Valid 979 个动作区间 Clip，输入为同一份冻结 ImageNet ResNet18 FP32 缓存。缓存保留原始真实帧索引、1–16 的真实长度、119 维标签和来源；模型仅消费 `[B,T,512]`，动态 Padding 不参与池化、BiGRU 有效序列和 Attention。ResNet18 被保留为可重建的官方冻结 Backbone 引用，正式 Epoch 的 CNN 调用数全部为 0。

三个独立模型仍为 Proposal 原结构：Model 1 为有效帧 Mean Pooling → Linear(512,119)；Model 2 为单层 BiGRU（每方向 hidden=128，输出 256 维）→ 有效时间 Mean Pooling → Linear(256,119)；Model 3 使用相同 BiGRU → 119 类各自的时间分数及 Masked Softmax → 类别对应加权聚合与线性分类。未增加 Spatial Attention 或其他模块。

## 锁定协议与独立初始化

完全复用 Phase 1.5 三份原始配置，未修改模型结构或训练协议：Seed 42、BCE、Batch 8、梯度累积 4、AdamW、峰值 LR 1e-3、Weight Decay 1e-4、梯度裁剪 5、前 2 Epoch Warmup、随后 Cosine 至峰值的 10%、最多 50 Epoch、Early Stopping Patience 10。每轮完整遍历 Train 一次，共 146 次更新，最后不足 32 个样本的累积窗口按实际样本数归一化。

三个模型各自从相同 ImageNet ResNet18 权重和 Seed 42 重新初始化，未继承诊断或其他模型的任务头。跨模型相同 Epoch 的输入顺序和目标 SHA256 相等。严格以 Valid 固定 31 类 mAP 选择最优轮，平分保留较早轮。最佳模型在完整 Valid 上从 0.05–0.95（步长 0.01）选择单一 Macro-F1 最优阈值，平分取较低阈值。

输出维度始终为 119。Defensive Rearing（ID 23）在 Train 无正例，因此 BCE 优化只包含其余 118 个类，类别专属参数行保持初始值，不作为正常动作发出；原始 119 维 Logits 仍保存。固定 31 主类本身不含该类别。BCE119 是包含该不可学习维度的原始诊断值；BCE118 对应实际优化目标。

## Valid 结果

| 模型 | 完成轮数 | 最佳轮 | Valid mAP31 | Valid Macro-F1 | BCE119 | BCE118 | Valid 阈值 | 总耗时/分钟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

本轮按既定主指标，Valid 最优为 **{best['model']}：{p(best['valid_mAP31'])} mAP**。这是一轮单随机种子的 Valid 比较，不能据此声称 Test 性能或显著性；未进行 Test 评估。

三模型正式顺序运行共 {runner_seconds/60:.2f} 分钟（包含模型/数据接口初始化及清理）；各模型内部记录的训练与最终导出总耗时之和为 {sum(r['wall_seconds'] for r in rows)/60:.2f} 分钟。整合开发、诊断和最终交付核验不计入训练耗时。

{chr(10).join(pair)}

## 曲线与资源

{curves}

若 Train Loss 持续下降而 Valid Loss 从最低点回升，说明继续拟合训练集没有同步改善概率泛化，存在过拟合迹象。BCE 最低轮与 mAP 最优轮可能不同：BCE 衡量概率拟合，mAP 衡量类别内排序；本轮严格使用已批准的 Valid mAP 选模，不改为按 BCE 选模。

Train 曲线为训练过程中更新前的 online 预测，汇集不同参数状态；Valid 为每轮结束后的固定模型。因此不能将两者当作同一固定 Checkpoint 的精确泛化差距。`history.csv`、JSON 和 JSONL 保存全部实际数值；每个模型的 `training_curves.svg/png` 提供 BCE118、BCE119、mAP31 三组曲线。历史 Macro-F1 固定阈值 0.5；上表 Macro-F1 使用最佳模型在 Valid 选出的阈值，二者含义不同。

| 模型 | 可训练参数张量元素 | GPU allocated峰值/MiB | GPU reserved峰值/MiB | 主进程RSS峰值/MiB | 主进程+Workers RSS峰值/MiB | Epoch中位耗时/秒 |
|---|---:|---:|---:|---:|---:|---:|
{resource}

GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU，训练为 FP32。显存统计来自 PyTorch，不包括驱动或桌面占用。RAM 每 20ms 采样本进程及其 DataLoader Workers 的 Windows Working Set；共享页面可能重复计算，不代表独占物理内存。每轮耗时包含 Train、Valid 和指标计算，不包含 Checkpoint/日志写入；总耗时包含它们、Worker 启动和最终 Valid 结果导出。分配/保留显存均按各模型重置后计量。

## 集成与最终核验

- Full Cache 整体 30,616 个已有文件通过 SHA256 预检，正式缓存 30,585 个文件、73,884 个唯一帧、5646 个 Clip，与锁定配置和上下文一致。
- 每个模型先使用 11 个真实 Train Clip、6 个真实 Valid Clip 验证有限 Loss、实际梯度和一次参数更新、不可学习类别保护、优化器与 RNG 恢复、恢复后相同 Logits。小样本不伪报完整 31 类 mAP；固定 31 类评价另用可验证的人工数值夹具检查。
- 正式 Best/Last Checkpoint、完整历史前缀、早停状态、调度、每轮 146 次更新、119 维标签、31 类混淆计数、阈值搜索和逐类 AP 均通过核验。使用已保存 Valid Logits 重新计算指标完全一致，不进行新的 Test 推理。
- 每个 Best Checkpoint 重新加载后，仅在 8 个现有 Valid 缓存样本上复核 Logits，最大误差见 `experiment_audit.json`；未运行 ResNet18。
- 保护核验通过：已有 {protection['historical_metadata_files']:,} 个文件元数据不变，{protection['historical_hashed_files']:,} 个历史/缓存文件内容 SHA256 不变；官方 ResNet18 权重 SHA256 不变。未打开受禁止的 Test 内容。旧大型任务权重核验元数据，正式缓存、Pilot 和非 Test 的 AnimalAct2 历史文件另核验内容。

## 文件与运行入口

- `comparison.csv` / `comparison.json`：三模型统一比较，含配置哈希、种子、阈值、时间、内存。
- `per_class_comparison_main31.csv`：固定类别映射、Train/Valid 支持数、三模型 AP/Precision/Recall/F1 及 AP 差值。
- `runs/modelN_bce_seed42/`：`best.pt`、`last.pt`、`config.json`、`provenance.json`、`history.csv/json/jsonl`、`training_curves.svg`、`results.json`、`threshold_selection.json`、`valid_predictions.pt` 和 `best_valid/` 下逐类/混淆表。
- `preflight/cache_and_protocol.json`、`integration_validation.json`、`experiment_audit.json`、`protection_verification.json`、`FINAL_VERIFICATION.json`：各阶段核验依据。
- `run_all.py` / `trainer.py`：新缓存适配入口，复用既有统一 Train/Valid Epoch、损失、调度、评价函数；`validate_integration.py` 为诊断入口，正式训练需先通过验证门禁。

Checkpoint 保存全部可学习任务头、优化器、调度器、早停、随机状态、历史和缓存/数据/实现指纹。完全冻结的 Backbone 不重复写入每个 Checkpoint，而由受 SHA256 保护的本地官方 ImageNet 权重重建。`last.pt` 支持正式中断恢复；已完成实验会跳过。诊断 Checkpoint 不得恢复到正式实验。

三个模型均已完成。当前停止并等待人工审核，不读取 Test，不启动第二轮或其他实验。
"""
    (OUT/"FINAL_FIRST_ROUND_REPORT.md").write_text(text,encoding="utf-8")


def main():
    enable_guard()
    rows,per_class,checks=audit()
    plot_curves()
    protection=verify()
    report(rows,per_class,protection)
    artifacts={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob("*"))
               if p.is_file() and p.name!="FINAL_VERIFICATION.json" and ".runtime_temp" not in p.parts}
    write_json(OUT/"FINAL_VERIFICATION.json",dict(status="passed",all_models_complete=True,experiment_audit=checks,protection=protection,
               files_excluding_this_manifest_and_runtime=len(artifacts),logical_bytes=sum(v["bytes"] for v in artifacts.values()),
               artifacts=artifacts,Test_content_read=False,second_round_started=False))
    print(json.dumps(dict(status="passed",report=str(OUT/"FINAL_FIRST_ROUND_REPORT.md"),models=rows),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
