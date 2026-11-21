# AnimalAct 2.0 第一轮正式实验

本目录独立整合 Phase 1.5 的三个 Proposal 模型与 Full Cache。正式数据、共享缓存、旧代码和历史结果只读。详细实际结果见 `FINAL_FIRST_ROUND_REPORT.md`，统一结果见 `comparison.csv`。

## 固定输入与协议

- 数据：`animalact2/release_v1/`，版本 `animalact2-2.0.0-release-v1`，仅 Train 4667 和 Valid 979 Clip。
- 缓存：`animalact2/phase1_5_cache_full/`，每 Clip 为真实变长 `[T,512]` FP32。真实帧数 1–16，不重复补帧。
- 三个模型完全复用 `phase1_5_1` 的结构；缓存直接进入 `forward_features`，Epoch 内禁止 ResNet18 前向。
- 相同 ImageNet 权重、Seed 42、BCE（118 个有 Train 支持的类，保留 119 维输出）、Batch 8、累积 4、AdamW LR 1e-3 / WD 1e-4 / Clip 5、2 Epoch Warmup + Cosine 至 10%、最多 50 Epoch、Patience 10。
- Valid 固定 31 主类 mAP 选模；最佳模型仅用 Valid 在 0.05–0.95、步长 0.01 上选单一 Macro-F1 阈值。没有 Test 输入或推理入口。
- Defensive Rearing（ID 23）没有训练正例，排除训练损失、恢复类别专属参数行、不正常发出预测；119 维原始 Logits 和 BCE119 诊断值仍保留。

## 入口

从项目根目录使用既有项目 `.venv`，无需新增依赖或下载权重：

```powershell
# 只读检查共享缓存与锁定配置；写入本目录的预检记录
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.first_round.preflight

# 少量真实缓存 Batch 验证，含参数更新、恢复和数值一致性；不是正式训练
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.first_round.validate_integration

# 正式入口：依次 Model 1、2、3；中断后运行同一命令从 last.pt 恢复
# 已完成实验跳过，诊断 checkpoint 不允许恢复为正式实验
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.first_round.run_all --execute-training

# 三模型完成后的结果核验及汇总，只读取已有 Valid 输出和少量 Valid 缓存
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.first_round.finalize
```

本轮完成后停止。上述命令用于说明已完成工作的入口，不代表批准自动开展下一轮实验。训练前必须通过当前代码/缓存指纹对应的集成验证。正式运行强制使用完整固定 Train/Valid，不提供采样、模型结构或超参数覆盖开关。

## 实现关系

- `trainer.FullCacheTrainer` 继承原统一 Trainer，复用原 `train_one_epoch` / `evaluate_loader`、优化器、损失、支持类别策略、调度、早停、指标与阈值函数。
- `phase1_5_cache_full.dataset.CachedFeatureDataset` 读取正式缓存并检查 Clip 键、帧数、数值和 SHA256。标签和来源由只读正式 Manifest 读取，不重建标签。
- `FeatureForwardView` 仅转换调用入口，没有新参数或新网络。Collate 的 `frames` 字段在本入口表示特征，由 `input_kind=frozen_resnet18_features` 区分；`lengths`、`valid_frame_mask`、`padding_mask` 保留原接口。
- `checkpoint.py` 沿用统一状态结构，适配独立输出路径和缓存来源指纹。
- `support.py` 的运行期保护禁止写入本目录之外、网络访问和原始视频/标注、旧任务权重及 Test 内容读取。仅进程内 TEMP/TMP 指向本目录，不改系统或全局环境变量。
- `validate_integration.py` 独立小样本验证；`finalize.py` 审核已保存结果、恢复少量 Valid 缓存预测、生成比较表及中文报告。

## 模型恢复与 Checkpoint

`best.pt`、`last.pt` 包含任务头全部权重、优化器、调度器、早停、随机状态、历史和来源指纹。冻结 Backbone 不重复存储，由既有本地官方权重重建；需要保留 `.cache/torch/checkpoints/resnet18-f37072fd.pth`。SHA256 为 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。

程序化恢复示例（仅已有 Train/Valid 缓存，不是原始视频推理）：

```python
import torch
from animalact2.first_round import checkpoint
from animalact2.first_round.support import OUT
from animalact2.phase1_5.models import build_model
from animalact2.phase1_5.losses import UnsupportedClassPolicy

path = OUT / "runs/model3_bce_seed42/best.pt"
saved = torch.load(path, map_location="cpu", weights_only=True)
model = build_model(saved["config"]).cuda().eval()
policy = UnsupportedClassPolicy(model, saved["class_policy"]["train_positive_counts"])
checkpoint.load(path, model, saved["config"], policy=policy)
# model.forward_features(features, lengths, valid_frame_mask) -> [B,119]
# 阈值在同目录 threshold_selection.json；不在 Test 上重新选择
```

## 交付文件

每个 `runs/modelN_bce_seed42/` 独立保存最佳/最后权重、完整配置、来源、逐轮 CSV/JSON/JSONL、训练曲线 SVG/PNG、最终 Valid 119 维输出及 31 类 AP/混淆统计。没有覆盖其他阶段。

- `comparison.csv/json`：完整 Valid 结果、阈值、时间、内存和配置哈希。
- `per_class_comparison_main31.csv`：固定 31 类及支持数、AP 与 P/R/F1、配对 AP 差。
- `experiment_audit.json`：状态、选模、恢复、指标和跨模型协议一致性检查。
- `protection_baseline.json` / `protection_verification.json`：历史保护基线及核验范围。
- `FINAL_VERIFICATION.json`：交付文件 SHA256、大小、总量与核验结论。

Train 曲线是 Epoch 内更新前 online 预测，Valid 为 Epoch 结束的固定模型。历史 F1 阈值为 0.5，最终 F1 使用最佳模型的 Valid 所选阈值。BCE119 包含不可学习类别，实际训练目标是 BCE118。本轮为区间 Clip 级、Seed 42 的第一轮结果，不能与旧版视频级数据直接比较。
