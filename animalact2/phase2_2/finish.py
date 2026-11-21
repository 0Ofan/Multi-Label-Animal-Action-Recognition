"""Independent saved-cache validation and Chinese pilot report, with no model run."""
from __future__ import annotations
import collections
import csv
import itertools
import json
from pathlib import Path
import time
import numpy as np
from animalact2.phase1_5_1.dataset import ReleaseClipDataset
from .support import ROOT,OUT,sha,digest,write_json


def inventory(directory):
    paths=[p for p in directory.rglob("*") if p.is_file() and ".runtime" not in p.parts]
    return dict(files=len(paths),logical_bytes=sum(p.stat().st_size for p in paths))


def main():
    started=time.perf_counter()
    plan=json.loads((OUT/"pilot_plan.json").read_text(encoding="utf-8"))
    ctx=json.loads((OUT/"cache/context.json").read_text(encoding="utf-8"))
    pilot=json.loads((OUT/"pilot_validation.json").read_text(encoding="utf-8"))
    extract=json.loads((OUT/"extraction_summary.json").read_text(encoding="utf-8"))
    heads=json.loads((OUT/"head_validation.json").read_text(encoding="utf-8"))
    geometry=json.loads((OUT/"adapter_validation.json").read_text(encoding="utf-8"))
    protection=json.loads((OUT/"protection_verification.json").read_text(encoding="utf-8"))
    assert all(r["status"]=="passed" for r in (pilot,extract,heads,geometry,protection))
    assert pilot["encoder_state_before"]==pilot["encoder_state_after"] and pilot["optimizer_updates"]==heads["optimizer_updates"]==0
    summary={};table=[];real_refs=0;slots=0;source_videos=set()
    for split,count in (("train",100),("valid",20)):
        original={r["clip_id"]:r for r in ReleaseClipDataset(split).records}
        rows=plan["records"][split];assert len(rows)==count
        expected_ids={r["clip_id"] for r in rows}
        assert {p.name for p in (OUT/"cache"/split).iterdir()}==expected_ids
        for r in rows:
            assert r==original[r["clip_id"]]
            d=OUT/"cache"/split/r["clip_id"]
            a=np.load(d/"features.npy",allow_pickle=False);receipt=json.loads((d/"receipt.json").read_text(encoding="utf-8"))
            length=r["effective_length"];lt=(length+1)//2
            assert a.shape==(lt,768) and a.dtype==np.float32 and np.isfinite(a).all()
            assert sha(d/"features.npy")==receipt["features_SHA256"]
            assert receipt["key"]==digest(receipt["descriptor"])
            assert receipt["descriptor"]["context_SHA256"]==digest(ctx)
            for k in ("clip_id","split","parent_video_id","video_path","frame_indices"):
                assert receipt["descriptor"][k]==r[k]
            assert receipt["descriptor"]["video_content_SHA256"]==sha(ROOT/r["video_path"])
            assert receipt["descriptor"]["label_and_provenance_SHA256"]==digest(dict(target=r["target_119"],evidence=r["label_evidence"]))
            assert receipt["target_119"]==r["target_119"] and receipt["label_evidence"]==r["label_evidence"]
            indices=iter(r["frame_indices"]);expected=[list(pair) for pair in itertools.zip_longest(indices,indices,fillvalue=None)]
            assert receipt["tubelet_frame_indices"]==expected
            assert [i for pair in expected for i in pair if i is not None]==r["frame_indices"]
            assert receipt["valid_tubelet_mask"]==[True]*lt
            assert receipt["tubelet_real_counts"]==[sum(v is not None for v in pair) for pair in expected]
            g=receipt["geometry"]
            assert g["real_length"]==length and g["real_hw"]==[224,398] and g["aligned_hw"]==[224,400]
            assert g["aligned_length"]==length+(length%2) and g["grid"]==[lt,14,25]
            real_refs+=length;slots+=lt;source_videos.add((split,r["parent_video_id"]))
            table.append(dict(split=split,clip_id=r["clip_id"],original_real_frames=length,feature_steps=lt,frame_coverage=True,label_provenance_equal=True,
                              dtype="float32",finite=True,all_real_source_indices_retained=True))
        summary[split]=dict(clips=count,parent_videos=len({r["parent_video_id"] for r in rows}),
                            length_counts=dict(sorted(collections.Counter(r["effective_length"] for r in rows).items())),
                            length1_3=sum(r["effective_length"]<=3 for r in rows),length4_15=sum(4<=r["effective_length"]<=15 for r in rows),
                            length16=sum(r["effective_length"]==16 for r in rows),odd_lengths=sum(r["effective_length"]%2 for r in rows),
                            single_frames=sum(r["effective_length"]==1 for r in rows),multilabel=sum(len(r["class_ids"])>1 for r in rows),
                            sampling_warning_clips=sum(bool(r["sampled_label_coverage_warnings"]) for r in rows))
    assert real_refs==extract["real_frames"] and slots==extract["feature_time_steps"]
    with (OUT/"independent_cache_checks.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    cache_stats=inventory(OUT/"cache")
    array_stats=sum(p.stat().st_size for p in (OUT/"cache").rglob("features.npy"))
    review=dict(status="reviewed",method="assistant visual inspection of generated full-field contact sheet; not bounding-box annotation or action certification",
                image="visualizations/geometry_contact_sheet.jpg",panels=8,
                observations=["CLZLJTLI frame13: cricket remains at the lower-left edge in both panels; source-edge truncation is not repaired.",
                              "GIJIFHSA frame134: full trees, perches and small-animal locations retain identical geometry for L1 and L16.",
                              "MPRQAPLS and YVMAHWZS: multiple horizontal animal positions remain in view.",
                              "PLRAOUPH: both birds and contact area remain; no temporal evidence is fabricated from the three-frame source.",
                              "Magenta marks only the two synthetic review columns; actual model padding is normalized zero."])
    write_json(OUT/"visual_review.json",review)
    result=dict(status="passed",selection=summary,real_frame_references=real_refs,feature_time_steps=slots,unique_parent_videos=len(source_videos),
                cache=cache_stats,array_bytes=array_stats,raw_feature_bytes=slots*768*4,independent_checks=120,
                all_arrays_and_label_sources_verified=True,Test_read=False,formal_training=False,seconds=time.perf_counter()-started)
    write_json(OUT/"independent_validation.json",result)
    htable="\n".join(f"| {h['head']} | {h['parameters']:,} | [8,119] | {h['CPU_CUDA_max_logit_error']:.3g} | {h['padded_feature_perturbation_max_logit_error']:.1f} | 通过 |" for h in heads["heads"])
    coverage="\n".join(f"| {s} | {v['clips']} | {v['parent_videos']} | {v['length1_3']} | {v['length4_15']} | {v['length16']} | {v['odd_lengths']} | {v['single_frames']} | {v['multilabel']} |" for s,v in summary.items())
    report=f"""# AnimalAct 2.0 Phase 2.2 VideoMAE小样本验证报告

**技术Pilot通过。仅完成100个Train、20个Valid Clip；停止等待审核。未进行正式训练、全量提取或Test读取。**

## 数据与抽样

数据版本 `animalact2-2.0.0-release-v1`，严格读取原始动作区间、既定真实帧索引、119维标签和原标注来源，不扩大时间范围、不改划分、不重新生成标签。样本为覆盖技术风险的目的性抽样，不代表总体类别或长度分布，不能用于报告模型性能。

| Split | Clip | 父视频 | 1–3帧 | 4–15帧 | 16帧 | 奇数帧Clip | 单帧Clip | 多标签Clip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{coverage}

两Split均覆盖真实长度1–16。包含已审查的边缘蟋蟀CLZLJTLI、小鸟GIJIFHSA、多位置水鸟MPRQAPLS、水禽YVMAHWZS、山羊CCKWWDSN、犬SMLRJXGD和整个源视频仅3帧的PLRAOUPH。具体Clip、选择理由、时间、帧索引、标签及annotation UID见 `sample_manifest.csv`、`pilot_plan.json`。

## 冻结编码器及输入适配

- 使用既有 `MCG-NJU/videomae-base-finetuned-kinetics`，revision `{pilot['source']['revision']}`，原文件SHA核验通过，没有下载或复制新权重。编码器 {pilot['encoder_parameters']:,} 参数完全冻结、eval、inference_mode、CUDA FP32 SDPA。
- 沿用release原RGB、PIL bilinear短边224等比Resize和ImageNet归一化，保留224×398完整画面；不使用中心裁剪、拉伸或方形Letterbox。
- 在Patch卷积内部将398宽对齐到400，仅右侧补2列归一化零值；不改变398列真实像素。图中的紫色仅用于标明这两列，实际模型输入不是紫色。
- 原始L为1–16，tubelet=2。仅当L为奇数时在模型内部添加一个归一化零值槽，实际索引记为null，不重复真实帧、不取区间外帧、不统一补到16。末尾含一个真实帧的tubelet仍为有效，长度为 `Lt=ceil(L/2)`，并保存 `tubelet_real_counts`。
- 按相同真实L组批，最大视频batch=2。Encoder不混入额外全空时间tubelet；分类头才按Lt动态Padding并使用Mask。
- 原生位置表按 `[8,14,14,768]` 解释，空间bicubic插值到14×25，时间linear插值到Lt，均为align_corners=False；原生8×14×14走精确原表。轴顺序单独验证，无宽高或时间轴交换。
- 输出 `[B,Lt,14,25,768]` 沿空间平均得到 `[B,Lt,768]`。未保留全部空间Token，也未缓存时序分类模块输出。

这是冻结预训练权重上的显式输入适配，不是原生固定尺寸接口。Kinetics分类器和fc_norm不参与编码，新分类头不使用AnimalAct1.0或ResNet18实验权重。

## 实际验收

1. 原生16帧、224×224随机输入：适配器与原始官方编码器最后层Token及空间池化特征完全一致，最大误差 **{geometry['native_16x224x224_max_token_error']:.1f}**。
2. 对1–16每种长度的全部136个真实帧位置做扰动，每帧最右下真实像素均改变且仅改变预期tubelet的右下Patch；单帧及奇数末帧均未被截断。
3. 对120个实际Clip核验：解码帧索引与release逐项一致，归一化真实像素在内部对齐前后逐位相同；原119维标签、label_evidence、原始标注引用一致。共 **{real_refs}** 个真实帧引用全部保留，生成 **{slots}** 个时序特征位置。每个时序位置都含真实帧，无全空特征位置。
4. 缓存独立复核120/120：FP32、维度、有限值、SHA256、Clip/父视频/Split、时间、标签来源、帧到tubelet映射、真实帧数量和有效Mask均通过。失败 **0**。
5. 位置编码坐标场测试确认时间、高、宽轴映射正确。完整画面对照图8组已查看，左下边缘动物及多横向位置保留；此为几何技术检查，不认证动作标注语义或模型注意力定位质量。

## 三个分类头接口及梯度

新代码只将输入维度扩展为768，保持Mean、单层BiGRU(hidden=128/方向)和类别特异时间Attention的研究结构，无额外投影或新模块。

| 分类头 | 参数量 | Train输出 | CPU/CUDA最大误差 | Padding扰动Logits误差 | 梯度 |
|---|---:|---|---:|---:|---|
{htable}

真实Train Batch采用原始帧长度 `[1,16,3,5,7,15,2,4]`，对应特征长度 `[1,8,2,3,4,8,1,2]`；无须按长度降序。Valid仅做4条前向检查，不参与梯度。三个头输出均有限；全部可训练参数张量均得到非零有限梯度，BiGRU正确使用packed sequence，Attention对无效时间步权重为0、有效时间权重和为1。

诊断使用Train支持的118类BCE，Defensive Rearing输出维度保留、其专属行损失梯度为0。本轮 **3次诊断性反向、0次优化器更新**，不保存新的任务头权重，不计算或报告小样本31类mAP/F1。

编码器前后状态SHA完全一致：`{pilot['encoder_state_before']}`。所有编码器参数requires_grad=False、grad=None，未发生任何微调。

## 实际时间、内存与磁盘

- 特征提取及120条缓存核验：**{extract['seconds']:.2f}秒**；120条新提取，0失败。包含解码、Resize、编码、FP32落盘及缓存核验，不含编码器加载和前置适配器单测。
- 完整Pilot程序（模型加载、适配器测试、提取、分类头验证、可视化）：**{pilot['program_seconds']:.2f}秒**。历史保护与最终报告整理另计，不属于训练时间。
- 提取PyTorch显存分配峰值：**{extract['CUDA_peak_allocated_bytes']/2**20:.2f}MiB**；保留峰值 **{extract['CUDA_peak_reserved_bytes']/2**20:.2f}MiB**。不含驱动或桌面占用。
- 提取阶段进程RSS峰值：**{extract['RAM']['parent_RSS_peak_bytes']/2**30:.3f}GiB**；完整Pilot程序峰值 **{pilot['overall_RAM']['parent_RSS_peak_bytes']/2**30:.3f}GiB**，20ms采样、无DataLoader子进程。
- 三头诊断显存分配峰值：{heads['CUDA_peak_allocated_bytes']/2**20:.2f}MiB（编码器已移回CPU，不混入该计量）。
- 纯特征数据：**{slots*768*4/2**20:.3f}MiB**；包含NPY头的数组文件：**{array_stats/2**20:.3f}MiB**；完整缓存目录含来源收据：**{cache_stats['logical_bytes']/2**20:.3f}MiB**，{cache_stats['files']}个文件。均为逻辑字节，不冒充Windows分配占用；全部新增文件大小及SHA另见最终清单。

缓存每Clip保存变长NPY，不在磁盘Padding。键包含预训练权重/版本、代码与预处理、源视频内容SHA、Clip/时间/帧索引、标签来源摘要。临时目录写入、fsync、原子目录提交；有效提交仅校验复用，不覆盖。当前范围锁定为120个Clip，未生成全量缓存。

## 局限与后续审核

- 已验证信息保留、数值和接口，不证明VideoMAE优于ResNet18，也不证明短动作可以可靠识别。
- 奇数末tubelet含一个真实帧和一个零值槽，与预训练的两帧分布不同；单帧没有运动过程。保留真实帧不等于恢复缺失的时间证据。
- 矩形/时间位置编码插值是适配选择，坐标轴正确不代表插值策略最优。398到400只占2列，不是把完整画面缩入224方形。
- 输出为tubelet时间特征，**不是独立逐帧特征**；末尾部分有效tubelet当前同样作为一个有效时间步，Mean按有效tubelet平均。真实帧计数保留，未自动引入额外加权实验。
- release原有短暂共标签未被选中帧命中的警告仍保留，本轮没有借窗口外帧、改标签或新增采样。

工程可行性已通过，RTX5060 Laptop 8GB可运行该冻结适配链路。是否批准全量Train/Valid缓存和后续正式对照，等待用户审核；本程序不会自动扩展范围或启动训练。

## 文件与保护

`adapter.py`：矩形/变长冻结编码器；`data.py`：120条抽样、可追溯缓存及Collate；`heads.py`：三个768维输入分类头；`tests.py`：数值/覆盖/Mask/梯度测试；`run_pilot.py`：限定范围入口；`finish.py`：独立缓存核验与报告。

依据文件：`adapter_validation.json`、`extraction_summary.json`、`head_validation.json`、`pilot_validation.json`、`clip_validation.csv`、`independent_cache_checks.csv`、`independent_validation.json`、`visualizations/geometry_contact_sheet.jpg`、`visual_review.json`、`protection_verification.json`。

保护复核：{protection['metadata_files']:,}个既有文件元数据和{protection['content_hashed_files']:,}个受保护文件内容SHA保持不变，包含正式Train/Valid数据、原权重和ResNet18第一轮交付。未读取Test视频、Test标注、Test Manifest或预测；未修改旧代码、模型、缓存或结果，未下载新权重，未开发前端。

最终文件清单排除运行时临时目录`.runtime/`及清单自身；缓存大小包含全部NPY、收据和context，不包含预训练权重或保护基线。没有删除历史文件。

**Phase2.2完成后停止，等待审核。**
"""
    (OUT/"FINAL_PHASE2_2_REPORT.md").write_text(report,encoding="utf-8")
    artifacts={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob("*"))
               if p.is_file() and ".runtime" not in p.parts and p.name!="FINAL_VERIFICATION.json"}
    write_json(OUT/"FINAL_VERIFICATION.json",dict(status="passed",files_excluding_runtime_and_this_manifest=len(artifacts),
                logical_bytes=sum(v["bytes"] for v in artifacts.values()),cache=cache_stats,independent_validation=result,
                protection=protection,artifacts=artifacts,Test_read=False,formal_training=False,full_cache_generated=False))
    print(json.dumps(dict(status="passed",report=str(OUT/"FINAL_PHASE2_2_REPORT.md"),cache=cache_stats,selection=summary),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
