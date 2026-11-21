"""Read-only final checks and Chinese cache report; no training or checkpoint operations."""
from __future__ import annotations
import ast
import json
import time
import torch
from .storage import CACHE,context
from .allocation import allocation_bytes
from .support import OUT,ROOT,sha,digest,enable_guard,write_json,verify


def main():
    begin=time.perf_counter();enable_guard()
    if (OUT/"FINAL_VERIFICATION.json").exists():raise FileExistsError("Preserve completed full-cache evidence")
    v=json.loads((OUT/"validation_summary.json").read_text(encoding="utf-8"))
    interfaces=json.loads((OUT/"reader_interface_checks.json").read_text(encoding="utf-8"))
    runs=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT/"extraction_runs").glob("run_*.json"))]
    assert v["status"]==interfaces["status"]=="passed" and v["verified_unique_frames"]==73884 and v["verified_clips"]==5646 and v["failures"]==0
    assert all(r["failure_count"]==0 for r in runs) and runs[-1]["new_CNN_unique_frames"]==0 and runs[-1]["decoded_parent_videos"]==0
    assert sum(r["new_CNN_unique_frames"] for r in runs)==72185
    assert sum(r["Pilot_reuse"]["independent_copied_frames_this_run"] for r in runs)==1699
    assert all(r["CNN_forward_calls"]==0 and not r["parameters_updated"] for r in interfaces["models"])
    ctx=context();assert ctx==json.loads((CACHE/"context.json").read_text(encoding="utf-8"))
    for p in OUT.rglob("*.py"):ast.parse(p.read_text(encoding="utf-8"),filename=str(p))
    cache_files=sorted(p for p in CACHE.rglob("*") if p.is_file())
    logical=sum(p.stat().st_size for p in cache_files);allocation=allocation_bytes(cache_files)
    sizes={name:sum(p.stat().st_size for p in (CACHE/name).rglob("*") if p.is_file()) for name in ("frames","clips")}
    protection=verify()
    generated=[r for r in runs if r["new_CNN_unique_frames"] or r["new_clips"]]
    generation_seconds=sum(r["elapsed_seconds"] for r in generated)
    peak_gpu=max(r["CUDA_peak_allocated_bytes"] for r in runs);peak_reserved=max(r["CUDA_peak_reserved_bytes"] for r in runs)
    peak_ram=max(r["RAM"]["parent_RSS_peak_bytes"] for r in runs)
    split_table="\n".join(f"| {r['split']} | {r['verified_clips']} | {r['parent_videos']} | {r['unique_frames']} | {r['clip_frame_references']} | {r['failed_clips']} |" for r in v["split_statistics"])
    report=f"""# AnimalAct 2.0 完整Train/Valid ResNet18共享缓存报告

**完整缓存生成和独立核验通过。未训练、未修改Checkpoint，现停止等待审核。**

## 1 实际完成数量

| Split | 成功Clip | 父视频 | 唯一帧 | Clip帧引用 | 失败Clip |
| --- | ---: | ---: | ---: | ---: | ---: |
{split_table}
| 合计 | 5646 | 2505 | 73884 | 81641 | 0 |

Train4667、Valid979，与release-v1正式顺序及标签一致。没有Test缓存或Test来源读取。每个唯一帧输出512维FP32；Clip保存真实`[T,512]`，T=1–16，不在磁盘上Padding、重复补帧或保存时序输出。

## 2 与Pilot完全相同的条件

复用Pilot的context、帧键/Clip键、label元数据摘要、数组读取和SHA校验函数；Full写入函数仅将输出边界改到新目录，沿用相同NPY、收据、`.pending_*`和原子目录提交格式。

- 官方权重：ImageNet ResNet18 IMAGENET1K_V1，文件SHA256 `{ctx['weights_SHA256']}`。
- context SHA256：`{digest(ctx)}`，与Pilot逐字段相同，包含代码/库版本。
- 完整画面、RGB、PIL bilinear、短边224等比Resize，当前398×224，ImageNet归一化，无中心裁剪/letterbox。
- 使用release-v1已确认的真实帧索引和变长规则；CNN冻结、eval、inference_mode，chunk8、FP32、TF32关闭。

先核对2505个源视频和对应原Train/Valid JSON的内容SHA。Pilot1699帧经SHA、身份键和上下文校验后独立复制到Full目录，不用软/硬链接、不改Pilot；只为另外**72185帧**运行CNN。共73884个唯一帧各有一份有效输出，没有重复提取已确认有效的帧。

## 3 断点续跑与完整性

每个分片/Clip由临时目录完整写数组、fsync及SHA收据，再原子提交。已有提交必须通过SHA、身份描述、形状、dtype和有限值检查才复用；未提交临时目录不当有效缓存，损坏提交明确报错，禁止静默覆盖。沿用单写入器机制，不提供并发写锁。

完成后已实际再次执行完整流程：新增CNN帧数**0**、视频解码**0**、新增Clip**0**，复用73884帧和5646个Clip；耗时**{runs[-1]['elapsed_seconds']:.2f}秒**。不为验证续跑重新计算已有特征。

独立核验逐个检查：

1. 全部帧库数组FP32、512维、无NaN/Inf、SHA/字节数与收据一致，73884个唯一键完整，无额外或重复键。
2. 全部5646个Clip形状等于真实长度，原帧顺序、Clip/视频/Split映射正确，每一行与共享帧库**逐位一致**。
3. 重新读取原Train/Valid JSON，按批准的终点截断及半开区间求标签交集，核对119维target、密集/采样支持、原Clip窗口、annotation UID/Pointer、动物名称及贡献交集。
4. 实际核验{v['positive_clip_class_pairs_checked']}个正Clip×类别组合、{v['original_annotation_contributor_links_checked']}条标签来源关系；31主类不变，Defensive Rearing训练正例仍为0。
5. Pilot对应120个Clip与Full缓存逐位相等，全部历史Pilot文件保持原状。

独立核验约**{v['elapsed_seconds']:.2f}秒**，无额外CNN推理。详细结果见`validation_summary.json`、`validation/clip_checks.csv`、`source_checks.csv`、`split_statistics.csv`、119类/31主类支持表及空的`errors.json`。

## 4 实际资源

- 构建总耗时：**{generation_seconds:.2f}秒，约{generation_seconds/60:.2f}分钟**，含源身份校验、Pilot复制、新帧提取和Clip组装；不含后续独立核验、完整复用检查及报告整理。
- GPU：{torch.cuda.get_device_name(0)}，总显存{torch.cuda.get_device_properties(0).total_memory/2**30:.2f}GiB。
- 提取PyTorch分配峰值：**{peak_gpu/2**20:.2f}MiB**，保留峰值{peak_reserved/2**20:.0f}MiB。不含驱动、桌面等其它显存。
- 提取进程RSS峰值：**{peak_ram/2**30:.2f}GiB**，Windows工作集20ms采样，不当作精确独占物理RAM。
- 缓存逻辑文件大小：**{logical:,}字节，{logical/2**20:.2f}MiB（{logical/2**30:.3f}GiB）**，共{len(cache_files)}个文件。
- Windows文件分配空间：**{allocation['file_allocation_bytes']:,}字节，{allocation['file_allocation_bytes']/2**20:.2f}MiB**。由FILE_STANDARD_INFO.AllocationSize逐文件读取，包含文件分配余量；不含目录索引/MFT等文件系统元数据，不能把它称为整个分区变化量。
- 其中帧库逻辑大小{sizes['frames']/2**20:.2f}MiB，Clip数组及收据{sizes['clips']/2**20:.2f}MiB；既有官方权重不重复复制。
- 生成失败{sum(r['failure_count'] for r in runs)}，最终核验失败{v['failures']}。

## 5 三模型读取接口

统一入口在本目录`dataset.py`：

```python
import torch
from torch.utils.data import DataLoader
from animalact2.phase1_5_cache_full.dataset import CachedFeatureDataset, collate_features
from animalact2.phase1_5_1.collate import move_batch_to_device

dataset = CachedFeatureDataset("train")  # Valid改为"valid"，Test不可用
loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=collate_features)
batch = move_batch_to_device(next(iter(loader)), "cuda")
with torch.no_grad():
    logits = model.forward_features(batch["frames"], batch["lengths"], batch["valid_frame_mask"])
```

model为原Model1、Model2或Model3；三者使用同一个Dataset和相同缓存。batch通用输入槽`frames`是`[B,Tmax,512]`，不是RGB；标签`[B,119]`。只在临时Batch做动态数值padding，BiGRU沿用真实lengths，Mean/Attention沿用mask。Clip ID、父视频、Split及完整来源仍可读取。

`FeatureForwardView(model)`复用Pilot无参数适配器，`make_feature_loader`按现有正式配置构建loader；没有新模型层，没有改训练参数。现有RGB训练入口未自动切换；后续正式采用缓存时需显式使用该接口，不把特征直接交给RGB forward。

{interfaces['automated_tests']}项接口测试通过；三个模型各读取同一个含Train/Valid长度1/10/16的6样本Batch，输出`[6,119]`且有限，mask与原数据一致。Model3无效时间Attention为0。检查期间CNN调用0、反向0、参数更新0，没有训练或写Checkpoint。

## 6 文件保护与停止

仅在`animalact2/phase1_5_cache_full/`写新文件。核对{protection['historical_metadata_files']}个旧文件大小/修改时间、{protection['historical_hashed_files']}个历史文件SHA（包含Pilot所有缓存）、官方ImageNet权重SHA，均未变。原始数据、release、模型、训练配置和Checkpoint不修改；Test视频、标注、Manifest及预测不打开，不下载依赖/模型。

最终全目录文件数/大小见`FINAL_VERIFICATION.json`，发布文件SHA清单见`artifact_manifest.json`。**完整缓存交付后停止等待审核，不启动正式训练、不修改Checkpoint、不自动进入下一阶段。**
"""
    (OUT/"FULL_CACHE_REPORT.md").write_text(report,encoding="utf-8")
    write_json(OUT/"storage_statistics.json",dict(logical_bytes=logical,cache_files=len(cache_files),parts=sizes,Windows_allocation=allocation))
    payload={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("artifact_manifest.json","FINAL_VERIFICATION.json")}
    write_json(OUT/"artifact_manifest.json",dict(files=payload,excluded_self_references=["artifact_manifest.json","FINAL_VERIFICATION.json"]))
    final=dict(status="complete_and_stopped_for_review",Train_clips=4667,Valid_clips=979,total_clips=5646,unique_frames=73884,clip_frame_references=81641,
               Pilot_valid_frames_independently_copied=1699,new_frames_inferred=72185,each_unique_frame_only_once=True,failures=0,
               cache_logical_bytes=logical,cache_file_allocation=allocation,cache_files=len(cache_files),generation_seconds=generation_seconds,
               verification_seconds=v["elapsed_seconds"],no_op_resume_seconds=runs[-1]["elapsed_seconds"],no_op_new_CNN_frames=0,
               GPU_peak_allocated_bytes=peak_gpu,GPU_peak_reserved_bytes=peak_reserved,RAM_peak_parent_RSS_bytes=peak_ram,
               context_identical_to_Pilot=True,FP32_uncompressed=True,real_lengths_no_disk_padding=True,all_arrays_keys_labels_sources_verified=True,
               interface_tests=interfaces["automated_tests"],models_reading_same_cache=3,model_parameters_updated=False,training_started=False,
               Test_read=False,Checkpoint_modified=False,model_or_formal_config_modified=False,protection=protection,
               finalization_seconds=time.perf_counter()-begin,stopped_for_review=True,automatic_next_stage=False,new_files=0,total_bytes=0,MiB=0)
    for _ in range(10):
        write_json(OUT/"FINAL_VERIFICATION.json",final)
        allfiles=[p for p in OUT.rglob("*") if p.is_file()];actual=len(allfiles),sum(p.stat().st_size for p in allfiles)
        if actual==(final["new_files"],final["total_bytes"]):break
        final["new_files"],final["total_bytes"]=actual;final["MiB"]=round(actual[1]/2**20,6)
    else:raise RuntimeError("Final file sizes did not stabilize")
    assert all(sha(OUT/path)==row["sha256"] for path,row in payload.items())
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
