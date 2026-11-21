"""Verify committed feature files and historical protection; write the Chinese report."""
from __future__ import annotations
import ast
import io
import json
import time
import unittest
import numpy as np
from .cache import CACHE,context,select_plan,selected_records,FrameStore,frame_key,clip_descriptor,read_array
from .support import OUT,enable_guard,write_json,sha,verify,digest
from . import tests


def mib(v):return v/2**20
def gib(v):return v/2**30


def main():
    started=time.perf_counter();enable_guard()
    if (OUT/"FINAL_VERIFICATION.json").exists():raise FileExistsError("Preserve the completed pilot")
    d=json.loads((OUT/"validation_results.json").read_text(encoding="utf-8"))
    fresh=json.loads((OUT/"fresh_process_cached_resources.json").read_text(encoding="utf-8"))
    assert d["status"]==fresh["status"]=="passed" and d["clips"]==dict(train=100,valid=20)
    assert not d["formal_training"] and not d["Test_read"] and fresh["no_raw_video_decode"]
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
    tested=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/"final_test_results.txt").write_text(stream.getvalue(),encoding="utf-8")
    assert tested.wasSuccessful() and tested.testsRun==7
    plan=select_plan();records=selected_records(plan);ctx=context();bank=FrameStore(CACHE,ctx)
    expected={frame_key(r,k,ctx) for r in records for k in r["frame_indices"]}
    assert set(bank.locations)==expected and len(expected)==1699
    for r in records:
        desc=clip_descriptor(r,ctx)
        array,receipt=read_array(CACHE/"clips"/r["split"]/r["clip_id"],dict(clip_descriptor=desc,clip_key=digest(desc)))
        assert array.shape==(r["effective_length"],512) and array.dtype==np.float32
        assert np.array_equal(array,np.stack([bank.get(key) for key in desc["frame_keys"]]))
    runs=d["extraction_runs"]
    assert [r["new_unique_frames_inferred"] for r in runs]==[68,1631,0] and runs[-1]["reused_clips"]==120
    files=[p for p in CACHE.rglob("*") if p.is_file()]
    assert len(files)==d["cache_file_count"] and sum(p.stat().st_size for p in files)==d["cache_total_file_bytes"]
    for p in OUT.rglob("*.py"):ast.parse(p.read_text(encoding="utf-8"),filename=str(p))
    protection=verify()
    e=d["full_cache_estimate"]
    creation=sum(r["elapsed_seconds"] for r in runs if r["new_unique_frames_inferred"])
    cmps={r["model"]:r for r in d["logits_comparisons"]};grads={r["model"]:r for r in d["gradient_comparisons"]}
    rows=[];numrows=[];resource_rows=[]
    for pair in d["speedup_pairs"]:
        name=pair["model"];a,b=[next(r for r in d["pilot_epoch_timings"] if r["model"]==name and r["input_mode"]==mode) for mode in ("realtime","cached")]
        rows.append(f"| {name} | {a['steady_pilot_epoch_seconds']:.4f} | {b['steady_pilot_epoch_seconds']:.4f} | {pair['pilot_steady_epoch_speedup']:.1f}× | {a['cold_start_pilot_epoch_seconds']:.2f} / {b['cold_start_pilot_epoch_seconds']:.2f} |")
        numrows.append(f"| {name} | {cmps[name]['maximum_absolute_logit_error']:.3e} | {grads[name]['maximum_absolute_gradient_difference']:.3e} | {pair['post_4step_maximum_parameter_difference']:.3e} |")
        resource_rows.append(f"| {name} | {mib(a['CUDA_peak_allocated_bytes']):.2f} / {mib(b['CUDA_peak_allocated_bytes']):.2f} | {gib(a['RAM']['parent_RSS_peak_bytes']):.2f} / {gib(b['RAM']['parent_RSS_peak_bytes']):.2f} | {gib(a['RAM']['parent_plus_workers_RSS_sum_peak_bytes']):.2f} / {gib(b['RAM']['parent_plus_workers_RSS_sum_peak_bytes']):.2f} |")
    report=f"""# AnimalAct 2.0 共享ResNet18缓存小样本验证报告

**验证通过，已停止等待审核。只生成100 Train、20 Valid Clip的缓存，没有全量缓存、Test或正式训练。**

## 1 范围与可追溯输入

使用release-v1，样本来自111个父视频。按真实帧长度分层：Train的1–3/4–15/16帧组为3/20/77，Valid为1/3/16；强制覆盖1/2/3/4/15帧及共享帧对，不按模型效果选样。固定seed42，见`selection.json`与`selected_manifest.jsonl`。

帧、119位标签、原Clip及标注来源均继承正式Manifest。复用现有准确帧索引解码、RGB、PIL bilinear、完整画面短边224及ImageNet归一化，不裁剪、不补帧。标签不写入特征数组，始终从只读Manifest取得。

## 2 缓存结构与构建结果

- 共120/120个Clip成功，失败0；1703个Clip帧引用对应1699个唯一帧，避免4次重复CNN提取。样本内重复帧较少，主要收益来自跨模型、跨epoch复用冻结CNN输出，不把31–38倍加速归因于这4帧。
- 帧库：每分片最多8个唯一帧，`[N,512]`；Clip文件：真实`[T,512]`，T=1–16，不保存padding。数组为未压缩`.npy` FP32，没有FP16、量化、NPZ或时序模块输出。
- 帧键包含视频ID/Split/源SHA、真实帧索引、预处理配置及Backbone权重SHA；上下文还含实现/库版本和数值模式。Clip键另外绑定帧序列、119标签及来源摘要。
- 每个数组有形状、dtype、字节数和SHA收据。读取检查完整描述与SHA，不因文件名相同就复用。源视频提取前核对内容SHA；读缓存时核对原Manifest及源视频stat，模型构造时再次验证官方ImageNet权重。
- 实际缓存共{d['cache_file_count']}个文件，逻辑字节数{d['cache_total_file_bytes']:,}，即**{mib(d['cache_total_file_bytes']):.2f} MiB**（帧库{mib(d['cache_bank_file_bytes']):.2f} MiB、Clip文件{mib(d['cache_clip_file_bytes']):.2f} MiB，另有context）。此为文件长度之和，不冒充NTFS簇/目录/MFT的精确物理占用。

两次建缓存合计**{creation:.2f}秒**，包括主动暂停后的重复索引检查及120条Clip组装/核验，不含Python导入或后续实时对照。提取峰值分配显存{mib(max(r['CUDA_peak_allocated_bytes'] for r in runs)):.2f} MiB，保留显存{mib(max(r['CUDA_peak_reserved_bytes'] for r in runs)):.0f} MiB，父进程RSS峰值{gib(max(r['RAM']['parent_RSS_peak_bytes'] for r in runs)):.2f} GiB。

## 3 中断续跑和完整性

1. 在68个已提交唯一帧处按分片边界主动中断。
2. 续跑复用68帧，只新增1631帧，完成全部120个Clip。
3. 再次运行复用1699帧及120个Clip，新增CNN计算0、视频解码0，耗时{runs[-1]['elapsed_seconds']:.2f}秒。

缓存构建合计恰好1699次唯一帧CNN计算。用于实时比较的额外CNN前向单独计时，不计入缓存构建，也不改写缓存。数组和收据在`.pending_*`目录写完后原子提交，未提交目录不会被当作有效分片；中断后只需重做未提交的工作。当前单写入器方式不提供多进程并发写锁。

7项测试通过，包含视频/帧/预处理/权重变化导致键变化、坏数组SHA拒绝、错上下文拒绝、120条dtype/长度/标签/来源逐条检查、短Batch动态Mask、Test和范围扩展拒绝，以及模拟未提交写入的恢复。测试损坏样例只在`test_fixtures/`，真实`cache/`未损坏。没有进行断电故障测试。

## 4 三模型数值与梯度

每个原模型分别对全部120条运行实时CNN和缓存特征，任务头初始化相同，比较119维logits。标签、父视频/Split、帧索引、lengths、Mask与来源逐条相同；容差预定为atol/rtol各2e-4。

| 模型 | 120条最大logits绝对误差 | 同批梯度最大绝对误差 | 一轮4次更新后最大参数差 |
| --- | ---: | ---: | ---: |
{chr(10).join(numrows)}

512维帧特征的最大差异为2.074e-5；两条路径的CNN批次分组不同，存在FP32舍入差异。NPY存储本身按FP32保留，帧库与每个Clip对应行逐位一致。当前结果支持FP32容差内等价，不声称全部训练轨迹逐位一致。

三个模型均使用已有`forward_features`，调用适配器没有新增层或参数，模型结构不变。CNN冻结/eval，BN状态和所有骨干参数不变；BiGRU、Attention及分类张量都有有限非零梯度并发生更新。Defensive Rearing专属行继续按既有118类训练目标策略保护，119维输出保持不变。

每条路径仅对100 Train、20 Valid做一轮诊断：13个Train microbatch、累积4、4次更新；Valid无反向。六条路径共24次真实小样本更新，独立资源复测另4次；没有正式数据集epoch、早停、阈值选择、最佳模型或任务权重保存。

## 5 读取与单轮耗时

缓存读取包括收据/SHA/形状检查，120条共{d['cache_read_benchmark']['seconds']:.3f}秒，约{d['cache_read_benchmark']['clips_per_second']:.1f} Clip/s，有效FP32 payload约{d['cache_read_benchmark']['payload_MiB_per_second']:.2f} MiB/s。该测试处于OS文件缓存已热状态，不能解释为SSD冷读带宽。

以下均使用未修改的正式配置：batch8、累积4、两个worker/loader、BCE、AdamW、首轮warmup LR5e-4、WD1e-4、seed42，两个loader均persistent workers。稳定时间包含输入、前向/反向、累积更新及Valid前向，不含完整mAP/AP导出和Checkpoint写入。

| 模型 | 实时稳定单轮秒 | 缓存稳定单轮秒 | 加速 | 含worker启动的实时/缓存秒 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

稳定阶段加速约31–38倍。小样本冷启动由两个loader的worker启动主导，不能把热身后的倍率套到每次进程启动；持续50轮训练的worker启动通常只发生一次。独立新进程缓存Model3的worker启动{fresh['worker_startup_seconds']:.2f}秒、首个诊断轮{fresh['steady_pilot_epoch_seconds']:.3f}秒，还包含首次CUDA/GRU运行开销，不能直接等同已热的0.22秒。

## 6 RAM与CUDA显存

以下列为“实时 / 缓存”。显存是PyTorch分配峰值，不包括驱动和桌面；RAM用20ms采样的Windows工作集/RSS。

| 模型 | GPU分配峰值MiB | 父进程RSS峰值GiB | 父进程+worker RSS峰值和GiB |
| --- | ---: | ---: | ---: |
{chr(10).join(resource_rows)}

上表是先实时、后缓存的长进程，缓存阶段仍保留此前CPU/pinned分配池，不能当作独立缓存内存需求。为此另开新进程仅跑缓存Model3：父进程峰值**{gib(fresh['RAM']['parent_RSS_peak_bytes']):.2f} GiB**，含worker的RSS峰值和**{gib(fresh['RAM']['parent_plus_workers_RSS_sum_peak_bytes']):.2f} GiB**，GPU分配峰值**{mib(fresh['CUDA_peak_allocated_bytes']):.2f} MiB**、保留峰值{mib(fresh['CUDA_peak_reserved_bytes']):.0f} MiB。

跨进程RSS和可能重复统计共享DLL/共享页，不是独占物理内存。没有安装psutil，使用Windows进程内存接口，仅查询本进程和自己的worker。当前验证保留原CNN对象及两个loader各两个worker，没有为提高缓存成绩改变正式配置。

## 7 全量Train/Valid估算（未生成）

本报告“全量”仅指Train4667+Valid979共5646条。通过只读Manifest统计得73884个唯一帧、81641个Clip帧引用；其它Clip只读取元数据用于估算，未解码或提取特征。Test未读取，因此不提供精确三Split容量。

- 唯一帧纯FP32 payload：{mib(e['unique_frame_FP32_payload_bytes']):.2f} MiB。
- 变长Clip纯FP32 payload：{mib(e['clip_FP32_payload_bytes']):.2f} MiB。
- 保留帧库、Clip文件、NPY头和收据的逻辑文件容量外推：**{mib(e['estimated_bank_plus_clip_files_bytes']):.1f} MiB，约{gib(e['estimated_bank_plus_clip_files_bytes']):.3f} GiB**。考虑NTFS簇/目录和余量，可先预留约0.5–0.6 GiB；并非已测得物理占用。
- 按唯一帧吞吐外推生成时间约**{e['estimated_generation_seconds_by_unique_frames']/60:.1f}分钟**；给出0.7–2倍的粗略范围，即约{e['estimated_generation_seconds_by_unique_frames']*.7/60:.1f}–{e['estimated_generation_seconds_by_unique_frames']*2/60:.1f}分钟。父视频长度、磁盘写入和分片数会影响实际值。
- 按当前稳定流水线外推完整Train+Valid每轮：实时约327–329秒，缓存约8.6–10.4秒；三个模型全部50轮名义时间实时约{e['estimated_three_models_50epoch_realtime_hours']:.2f}小时、缓存约{e['estimated_three_models_50epoch_cached_hours']:.2f}小时，另加建缓存、worker启动、完整评价/保存及系统负载。完整训练没有实际执行，不能把此数当作承诺；与Phase1.5较小基准的外推不同。

主要结论：在当前完全冻结ResNet18、固定真实帧与预处理条件下，共享缓存通过准确性、训练接口和资源验证，值得在审核后扩展到Train/Valid。未来若更改图像增强/帧索引/预处理或微调CNN，必须用匹配的新键重建对应特征，不能继续复用当前缓存。

## 8 保护与停止

历史保护通过：{protection['historical_metadata_files']}个旧文件大小/修改时间、{protection['hashed_non_Test_non_tensor_files']}个非Test非张量历史文件SHA及官方ImageNet权重SHA未变。release-v1、原始视频、旧模型及正式训练配置未修改；未下载依赖/模型，未读取Test视频、标注、清单或预测。原数据标签、Mask及31主类定义不变。

最终文件/字节统计见`FINAL_VERIFICATION.json`，逐文件SHA见`artifact_manifest.json`。**到此停止，等待审核；不自动生成全量缓存或开始正式训练。**
"""
    (OUT/"CACHE_PILOT_REPORT.md").write_text(report,encoding="utf-8")
    inventory={p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("artifact_manifest.json","FINAL_VERIFICATION.json")}
    write_json(OUT/"artifact_manifest.json",dict(files=inventory,excluded_self_references=["artifact_manifest.json","FINAL_VERIFICATION.json"]))
    final=dict(status="completed_and_stopped_for_review",Train_clips=100,Valid_clips=20,unique_parent_videos=111,unique_frames=1699,clip_frame_references=1703,cache_failures=0,
               FP32_only=True,compression=False,variable_length_clip_arrays=True,frame_shards_unique=True,automated_tests_passed=7,
               per_model_logits_compared_clips=120,all_three_models_logits_within_FP32_tolerance=True,gradient_checks_passed=True,
               cache_creation_CNN_evaluations=1699,no_op_resume_CNN_evaluations=0,diagnostic_updates=28,formal_training_started=False,temporal_outputs_or_task_weights_saved=False,
               cache_files=len(files),cache_bytes=d["cache_total_file_bytes"],cache_MiB=mib(d["cache_total_file_bytes"]),extraction_seconds=creation,
               full_Train_Valid_estimate=e,protection=protection,Test_read=False,full_cache_generated=False,formal_configs_or_models_changed=False,
               fresh_process_cached_RAM=fresh["RAM"],fresh_process_cached_peak_GPU_bytes=fresh["CUDA_peak_allocated_bytes"],
               finalization_seconds=time.perf_counter()-started,stopped_for_review=True,automatic_followup=False,new_files=0,total_bytes=0,MiB=0)
    for _ in range(10):
        write_json(OUT/"FINAL_VERIFICATION.json",final)
        allfiles=[p for p in OUT.rglob("*") if p.is_file()];actual=len(allfiles),sum(p.stat().st_size for p in allfiles)
        if actual==(final["new_files"],final["total_bytes"]):break
        final["new_files"],final["total_bytes"]=actual;final["MiB"]=round(actual[1]/2**20,6)
    else:raise RuntimeError("Could not finalize file sizes")
    assert all(sha(OUT/path)==value["sha256"] for path,value in inventory.items())
    print(json.dumps({k:v for k,v in final.items() if k not in ("full_Train_Valid_estimate",)},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
