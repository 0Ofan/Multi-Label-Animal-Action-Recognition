# AnimalAct 2.0 Phase 2.4 VideoMAE第一轮正式训练报告

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
| mean | 39 | 29 | 43.11% | 46.05% | 0.071370 | 0.054724 | 0.32 | 3.26 |
| bigru_mean | 50 | 41 | 45.85% | 45.94% | 0.065240 | 0.057917 | 0.07 | 4.21 |
| bigru_attention | 31 | 21 | 46.95% | 45.34% | 0.055903 | 0.049656 | 0.12 | 2.77 |

本轮按预设主指标最优为 **bigru_attention，Valid mAP 46.95%**。这是单随机种子、同一Valid选模结果，不是Test泛化结论或统计显著性结论。

按Macro-F1排序，最高为 **mean：46.05%**。mAP与F1排序不同，继续沿用原定mAP主指标选模，完整保留三个候选及各自Valid阈值。

三头顺序运行总耗时约 **10.42分钟**，包括初始化和清理；各模型内部训练/Valid/保存/最终导出累计 **10.23分钟**。整合开发、前置诊断、最终核验及报告不计入训练耗时。

## 与已有ResNet18第一轮比较

下表为VideoMAE减去相同头结构的ResNet18，百分点用于mAP/F1，BCE为绝对差值：

| 分类头 | mAP变化/百分点 | Macro-F1变化/百分点 | BCE118变化 | 31类AP提升/下降/相同 |
|---|---:|---:|---:|---|
| mean | +3.16 | +3.25 | +0.009596 | 24 / 7 / 0 |
| bigru_mean | +4.41 | +2.88 | +0.006912 | 19 / 12 / 0 |
| bigru_attention | +4.24 | +2.03 | -0.001228 | 22 / 9 / 0 |

完整六模型表见 `backbone_comparison.csv`，逐类对照见 `backbone_per_class_main31.csv`。标签映射、Valid正例支持数及评价规则逐项核验一致，只读取已有ResNet18 Valid汇总和历史，不读取旧预测或Test。

比较属于**冻结预训练Backbone方案比较**：ImageNet与Kinetics预训练不同，特征512→768、时间步1–16→1–8，分类头输入参数量随之变化。不能将差值归因于单独某一网络结构，也不能直接与AnimalAct1.0视频级mAP比较。

## 曲线、过拟合与资源

- mean：最佳轮Train online mAP 95.57%，Valid 43.11%；Valid BCE118最低点为第3轮（0.045508），最终为0.056172；最终Train online BCE118为0.012194。停止原因：early_stopping_patience10。
- bigru_mean：最佳轮Train online mAP 99.93%，Valid 45.85%；Valid BCE118最低点为第9轮（0.042670），最终为0.059623；最终Train online BCE118为0.002010。停止原因：max_epochs50。
- bigru_attention：最佳轮Train online mAP 99.26%，Valid 46.95%；Valid BCE118最低点为第7轮（0.042355），最终为0.056576；最终Train online BCE118为0.002811。停止原因：early_stopping_patience10。

Train Loss下降、Valid Loss从最低点回升时，提示概率泛化存在过拟合。BCE最小轮与mAP最优轮不必一致：前者衡量概率拟合，后者衡量类别内排序。本轮不因观察结果而改变既定选模规则。

Train mAP/F1/Loss为Epoch内更新前的online混合参数状态，Valid是Epoch结束的固定模型，两者不是同一固定Checkpoint的精确泛化差距。历史F1阈值0.5；上表最终F1使用最佳模型的Valid所选阈值。`training_curves.png/svg`展示BCE118、BCE119和mAP31，完整数值见history CSV/JSON/JSONL。

| 分类头 | 任务头参数 | GPU分配峰值/MiB | GPU保留峰值/MiB | 主进程RSS/MiB | 主进程+Workers RSS/MiB | Epoch中位耗时/秒 |
|---|---:|---:|---:|---:|---:|---:|
| mean | 91,511 | 66.1 | 68.0 | 1840.3 | 4455.1 | 4.45 |
| bigru_mean | 720,247 | 98.7 | 108.0 | 1945.1 | 4555.3 | 4.60 |
| bigru_attention | 750,711 | 99.2 | 112.0 | 2016.9 | 4628.1 | 4.72 |

CUDA设备为RTX5060 Laptop GPU。显存为PyTorch分配/保留峰值，不含驱动及桌面。RAM为Windows Working Set，20ms采样；主进程+Workers之和可能重复计算共享页，不等于独占物理RAM。每轮耗时含Train、Valid和指标，不含Checkpoint/日志I/O；总模型耗时包含它们及最终Valid导出。

## 整合与完成核验

- 训练前校验Full Cache **11,323个交付文件SHA256**，包含全部5646份数组/收据及范围/来源说明。
- 每个头先对11个真实Train Clip完成一次诊断更新，完整979个Valid上运行评价和阈值搜索；Loss有限、各参数张量更新、ID23参数行固定、Padding扰动不改变Logits。诊断分数不是正式结果，也未用于正式选模。
- Checkpoint恢复后，完整Valid Logits逐位一致；AdamW、调度、早停以及Python/NumPy/Torch CPU/CUDA随机状态恢复。诊断Checkpoint拒绝初始化正式运行。
- 最终核验Best/Last状态、完整历史前缀、严格最佳轮、早停、逐轮146次更新、学习率、Clip顺序、119维真实标签、31类AP与2×2混淆统计。保存的Valid Logits重算mAP/F1/BCE和阈值，与results.json完全一致。
- 最佳Checkpoint另用8条Valid缓存复核恢复Logits，并检查真实长度1/3/4/16对应特征长度1/2/2/8的Padding不变性和Attention Mask。结果见 `experiment_audit.json`，不调用编码器。
- 历史保护：79,312个旧文件元数据、11,769个受保护文件内容SHA复核不变，包含release、Full Cache、Pilot和ResNet18第一轮。旧Checkpoint仅在保护工具中按字节哈希，不作为本轮模型载入。

## 交付与停止

- `runs/modelN_bce_seed42/`：Best/Last Checkpoint、完整配置、来源、全部历史、曲线、results.json、Valid原始Logits/target/Clip ID、阈值搜索，以及119类/31类AP和混淆表。
- `comparison.csv/json`：三个VideoMAE分类头统一结果、资源、阈值、配置哈希。
- `per_class_comparison_main31.csv`：三头逐类AP、Precision/Recall/F1及配对变化。
- `backbone_comparison.csv`、`backbone_paired_changes.csv`、`backbone_per_class_main31.csv`：现有ResNet18与新VideoMAE的Valid对照。
- `preflight/`、`integration_validation.json`、`experiment_audit.json`、`backbone_comparison_checks.json`、`protection_verification.json`、`artifact_manifest.json`、`FINAL_VERIFICATION.json`：核验依据和SHA清单。

Checkpoint包含全部可学习头权重、AdamW、调度、早停、RNG、历史及完整缓存/数据/实现指纹；冻结编码器只保存来源引用。`run_all --execute-training` 对未完成实验使用last.pt恢复，已完成则跳过，禁止覆盖旧实验。

全部新代码和结果仅在 `animalact2/phase2_4/`。没有修改ResNet18/Pilot/Full Cache，没有读取Test视频、标注、Manifest或预测，没有ASL、微调、额外消融、特征提取、模型下载或前端开发。

**三个分类头全部完成后停止，等待人工审核，不自动进行Test评价或下一轮实验。**
