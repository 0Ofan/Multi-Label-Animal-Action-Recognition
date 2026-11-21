# AnimalAct 2.0 VideoMAE完整Train/Valid缓存

固定release-v1 Train4667、Valid979；输出为每Clip变长 `[Lt,768]` FP32，`Lt=ceil(真实选中帧数/2)`，不是逐帧CNN特征。真实帧、标签、标注来源、Mask及半有效末tubelet均保留。

**直接复用Phase2.2原始冻结适配器、预处理、位置插值、奇数帧处理、Context和缓存键**。Full仅增加独立存储路径、全量调度和核验，不改变模型。原生VideoMAE权重本地读取，无网络下载；全部Pilot及历史结果只读。

每个 `cache/train或valid/ClipID/` 下有未压缩FP32 `features.npy` 和带SHA256的 `receipt.json`。全部空间Token和RGB不落盘。有效缓存拒绝覆盖，临时目录写入并fsync后原子提交；单写入器，勿并行运行多个生成入口。

## 执行入口

从项目根目录，使用既有项目环境：

```powershell
# 建立一次历史保护基线，重复调用不会覆盖
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_3_full_cache.support

# 生成；中断后运行相同命令复用已通过的原子提交缓存
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_3_full_cache.extract

# 逐Clip独立验证，无模型构造和视频解码
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_3_full_cache.validate

# 已完整缓存的复用检查，禁止新提取，编码器不构造
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_3_full_cache.extract --resume-check

# 历史/Pilot只读SHA与元数据复核，旧Checkpoint仅按字节哈希不加载
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_3_full_cache.support verify

# 生成报告和最终清单，不执行模型
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_3_full_cache.finalize
```

已完成的summary和交付不允许覆盖；这些命令说明本次工程入口，不表示批准下一轮训练。出现损坏提交会明确报错，不静默删除或覆盖。未提交`.pending_*`不能算有效缓存，需显式检查。

## 读取接口

```python
from torch.utils.data import DataLoader
from animalact2.phase2_3_full_cache.dataset import CachedFeatureDataset, collate_features

ds = CachedFeatureDataset("valid")
batch = next(iter(DataLoader(ds, batch_size=8, collate_fn=collate_features)))
# features: [B,Lt_max,768]; lengths: [B]; valid_frame_mask/padding_mask: [B,Lt_max]
# target: [B,119]; clip_ids / parent_video_ids / real_lengths
```

Dataset仅启用Train/Valid，默认要求完整验证成功。单条样本另含原始 `frame_indices`、`tubelet_frame_indices`、`tubelet_real_counts` 和完整 `metadata`。后续可直接接Phase2.2的三个分类头；本阶段不运行模型分类头，不计算mAP或选择阈值，不进行任何训练。

## 资源与证据

实际资源、成功/失败和保护结论见 `FULL_VIDEOMAE_CACHE_REPORT.md`。`extraction_summary.json` 保存完成全量缓存的成功生成进程（本次为补齐段），`runs/` 保留完整落盘的运行记录；`resume_check.json` 是全量复用实测。`validation_summary.json`、逐Clip/逐类/分区CSV和 `reader_interface_checks.json` 提供核验依据。

本次存在一次原进程最终摘要未落盘的中断。`interruption_recovery.json` 保存原始最后进度及5388份有效提交的逐项核验；之后仅补齐剩余258份，`extraction_summary.json`记录这次成功补齐段。报告明确区分累计耗时下限与续跑段的GPU/RAM峰值，不将无法追溯的中断前峰值编造成全程测量。

`storage_statistics.json` 分别记录逻辑大小和Windows实际文件分配空间；后者不含MFT/目录索引。`artifact_manifest.json` 覆盖交付文件但排除自身、最终验证文件及临时运行目录；`FINAL_VERIFICATION.json` 单独保存清单SHA，避免自引用。

完成后停止等待审核。不读取Test，不修改ResNet18/Pilot/正式数据，不自动开始训练。
