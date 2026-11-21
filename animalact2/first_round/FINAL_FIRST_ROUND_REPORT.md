# AnimalAct 2.0 第一轮正式实验报告

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
| model1 | 45 | 35 | 39.95% | 42.79% | 0.048355 | 0.045128 | 0.17 | 5.71 |
| model2 | 50 | 42 | 41.44% | 43.06% | 0.059456 | 0.051005 | 0.21 | 6.48 |
| model3 | 35 | 25 | 42.72% | 43.31% | 0.057855 | 0.050885 | 0.11 | 4.61 |

本轮按既定主指标，Valid 最优为 **model3：42.72% mAP**。这是一轮单随机种子的 Valid 比较，不能据此声称 Test 性能或显著性；未进行 Test 评估。

三模型正式顺序运行共 17.00 分钟（包含模型/数据接口初始化及清理）；各模型内部记录的训练与最终导出总耗时之和为 16.80 分钟。整合开发、诊断和最终交付核验不计入训练耗时。

- model2 相对 model1：Valid mAP +1.49 个百分点，Macro-F1 +0.26 个百分点；逐类 AP 提升 21 类、下降 10 类、相同 0 类。
- model3 相对 model1：Valid mAP +2.77 个百分点，Macro-F1 +0.52 个百分点；逐类 AP 提升 23 类、下降 8 类、相同 0 类。
- model3 相对 model2：Valid mAP +1.28 个百分点，Macro-F1 +0.25 个百分点；逐类 AP 提升 14 类、下降 17 类、相同 0 类。

## 曲线与资源

- `model1`：最佳轮 Train online mAP 77.61%，Valid mAP 39.95%；Valid BCE118 最低点在第 14 轮（0.044518），最终为 0.045439；最终 Train online BCE118 0.023213；停止原因 `early_stopping_patience10`。
- `model2`：最佳轮 Train online mAP 95.90%，Valid mAP 41.44%；Valid BCE118 最低点在第 18 轮（0.043822），最终为 0.052755；最终 Train online BCE118 0.009222；停止原因 `max_epochs50`。
- `model3`：最佳轮 Train online mAP 94.28%，Valid mAP 42.72%；Valid BCE118 最低点在第 9 轮（0.044121），最终为 0.057223；最终 Train online BCE118 0.007260；停止原因 `early_stopping_patience10`。

若 Train Loss 持续下降而 Valid Loss 从最低点回升，说明继续拟合训练集没有同步改善概率泛化，存在过拟合迹象。BCE 最低轮与 mAP 最优轮可能不同：BCE 衡量概率拟合，mAP 衡量类别内排序；本轮严格使用已批准的 Valid mAP 选模，不改为按 BCE 选模。

Train 曲线为训练过程中更新前的 online 预测，汇集不同参数状态；Valid 为每轮结束后的固定模型。因此不能将两者当作同一固定 Checkpoint 的精确泛化差距。`history.csv`、JSON 和 JSONL 保存全部实际数值；每个模型的 `training_curves.svg/png` 提供 BCE118、BCE119、mAP31 三组曲线。历史 Macro-F1 固定阈值 0.5；上表 Macro-F1 使用最佳模型在 Valid 选出的阈值，二者含义不同。

| 模型 | 可训练参数张量元素 | GPU allocated峰值/MiB | GPU reserved峰值/MiB | 主进程RSS峰值/MiB | 主进程+Workers RSS峰值/MiB | Epoch中位耗时/秒 |
|---|---:|---:|---:|---:|---:|---:|
| model1 | 61,047 | 108.6 | 130.0 | 1844.5 | 4458.2 | 7.09 |
| model2 | 523,639 | 138.5 | 156.0 | 1950.6 | 4562.9 | 7.29 |
| model3 | 554,103 | 138.9 | 158.0 | 2032.1 | 4636.3 | 7.24 |

GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU，训练为 FP32。显存统计来自 PyTorch，不包括驱动或桌面占用。RAM 每 20ms 采样本进程及其 DataLoader Workers 的 Windows Working Set；共享页面可能重复计算，不代表独占物理内存。每轮耗时包含 Train、Valid 和指标计算，不包含 Checkpoint/日志写入；总耗时包含它们、Worker 启动和最终 Valid 结果导出。分配/保留显存均按各模型重置后计量。

## 集成与最终核验

- Full Cache 整体 30,616 个已有文件通过 SHA256 预检，正式缓存 30,585 个文件、73,884 个唯一帧、5646 个 Clip，与锁定配置和上下文一致。
- 每个模型先使用 11 个真实 Train Clip、6 个真实 Valid Clip 验证有限 Loss、实际梯度和一次参数更新、不可学习类别保护、优化器与 RNG 恢复、恢复后相同 Logits。小样本不伪报完整 31 类 mAP；固定 31 类评价另用可验证的人工数值夹具检查。
- 正式 Best/Last Checkpoint、完整历史前缀、早停状态、调度、每轮 146 次更新、119 维标签、31 类混淆计数、阈值搜索和逐类 AP 均通过核验。使用已保存 Valid Logits 重新计算指标完全一致，不进行新的 Test 推理。
- 每个 Best Checkpoint 重新加载后，仅在 8 个现有 Valid 缓存样本上复核 Logits，最大误差见 `experiment_audit.json`；未运行 ResNet18。
- 保护核验通过：已有 67,639 个文件元数据不变，31,970 个历史/缓存文件内容 SHA256 不变；官方 ResNet18 权重 SHA256 不变。未打开受禁止的 Test 内容。旧大型任务权重核验元数据，正式缓存、Pilot 和非 Test 的 AnimalAct2 历史文件另核验内容。

## 文件与运行入口

- `comparison.csv` / `comparison.json`：三模型统一比较，含配置哈希、种子、阈值、时间、内存。
- `per_class_comparison_main31.csv`：固定类别映射、Train/Valid 支持数、三模型 AP/Precision/Recall/F1 及 AP 差值。
- `runs/modelN_bce_seed42/`：`best.pt`、`last.pt`、`config.json`、`provenance.json`、`history.csv/json/jsonl`、`training_curves.svg`、`results.json`、`threshold_selection.json`、`valid_predictions.pt` 和 `best_valid/` 下逐类/混淆表。
- `preflight/cache_and_protocol.json`、`integration_validation.json`、`experiment_audit.json`、`protection_verification.json`、`FINAL_VERIFICATION.json`：各阶段核验依据。
- `run_all.py` / `trainer.py`：新缓存适配入口，复用既有统一 Train/Valid Epoch、损失、调度、评价函数；`validate_integration.py` 为诊断入口，正式训练需先通过验证门禁。

Checkpoint 保存全部可学习任务头、优化器、调度器、早停、随机状态、历史和缓存/数据/实现指纹。完全冻结的 Backbone 不重复写入每个 Checkpoint，而由受 SHA256 保护的本地官方 ImageNet 权重重建。`last.pt` 支持正式中断恢复；已完成实验会跳过。诊断 Checkpoint 不得恢复到正式实验。

三个模型均已完成。当前停止并等待人工审核，不读取 Test，不启动第二轮或其他实验。
