# Phase 2.5 多种子验证与过拟合分析

已复用Phase2.4 Seed42三组结果，仅补充Seed123、2026六组。结构、缓存、BCE118、AdamW、LR1e-3、WD1e-4、Batch8×累积4、2轮Warmup+Cosine至10%、最多50轮、Patience10、Valid31类mAP选模与Valid阈值规则均不变。无增强、微调、特征提取或Test读取。

## 三种子汇总

均值±样本标准差（ddof=1，N=3），指标为最佳Checkpoint固定权重评价。

| 头 | Valid mAP | Valid Macro-F1 | Valid BCE118 | Train−Valid mAP差/百分点 |
|---|---|---|---|---|
| mean | 43.15 ± 0.57% | 45.17 ± 0.83% | 0.055604 ± 0.000762 | 53.59 |
| bigru_mean | 46.28 ± 0.37% | 46.53 ± 0.51% | 0.056457 ± 0.002738 | 53.64 |
| bigru_attention | 46.78 ± 0.92% | 46.20 ± 1.00% | 0.052223 ± 0.004309 | 52.91 |

按Valid mAP均值，首选bigru_attention。三种子只反映本次重复性，不证明统计显著性或Test性能。

## 每种子固定Checkpoint结果

两Split使用同一个最佳权重和该权重的Valid所选阈值；Train不重新选阈值。

| 头 | Seed | 最佳/完成轮 | Train mAP | Valid mAP | Train F1 | Valid F1 | Train/Valid BCE118 | 最低Valid BCE轮 |
|---|---|---|---|---|---|---|---|---|
| mean | 42 | 29/39 | 96.45% | 43.11% | 92.35% | 46.05% | 0.01260/0.05472 | 3 |
| bigru_mean | 42 | 41/50 | 99.95% | 45.85% | 94.75% | 45.94% | 0.00229/0.05792 | 9 |
| bigru_attention | 42 | 21/31 | 99.61% | 46.95% | 92.78% | 45.34% | 0.00484/0.04966 | 7 |
| mean | 123 | 37/47 | 96.90% | 42.60% | 92.43% | 45.06% | 0.01188/0.05606 | 3 |
| bigru_mean | 123 | 30/40 | 99.83% | 46.55% | 95.20% | 46.80% | 0.00347/0.05330 | 7 |
| bigru_attention | 123 | 19/29 | 99.49% | 45.78% | 91.89% | 45.96% | 0.00530/0.04982 | 6 |
| mean | 2026 | 40/50 | 96.88% | 43.74% | 89.12% | 44.41% | 0.01185/0.05603 | 3 |
| bigru_mean | 2026 | 49/50 | 99.98% | 46.43% | 94.89% | 46.86% | 0.00189/0.05816 | 8 |
| bigru_attention | 2026 | 37/47 | 99.97% | 47.60% | 93.91% | 47.30% | 0.00193/0.05720 | 8 |

完整BCE119、阈值、资源、耗时及差距见all_seed_results.csv。训练Online指标是Epoch内混合参数，不能冒充这里的固定Checkpoint Train指标。六组history.csv及training_curves.png/svg独立保留；固定Train/Valid预测及指标在fixed_checkpoint/，Seed42原文件不修改。

## 过拟合判读

固定Checkpoint的Train/Valid差距比Online曲线更适合判断泛化。结合表中差距、Train Loss下降与Valid Loss回升，以及最佳mAP轮晚于最低Valid BCE轮，说明继续优化排序时概率拟合可能已恶化。mAP和BCE目标不同，本轮按原协议选模，不事后改选较早Checkpoint。

31类逐类AP和跨种子标准差见per_class_AP.csv与per_class_AP_stability.csv；支持数随种子相同，较大AP标准差应结合少量正例解读，不能仅按单次AP提升选配置。

长度诊断分为真实1–3、4–15和16帧，而非tubelet长度。见length_and_coverage_groups.csv：若组内缺少主类正例，full31_mAP为空；supported_main_mAP仅为该组有正例类别的均值，不能与完整31类mAP直接比较或用于选模。表中同时保存支持类别数与缺失ID，Macro-F1仍按固定31类定义。

标签覆盖警告标签本身在未命中帧中有密集支持，但输入采样未命中。coverage_warning_labels.csv仅统计这些正标签的Recall和分数，不为全为正例的标签集合计算误导性AP/Precision。该诊断是关联观察：不能据此认定警告样本所有错误均由采样造成，也不修改标签或输入。

ResNet_comparison.csv对照ResNet第一轮单Seed42与VideoMAE三种子汇总；重复次数不对等，不代表ResNet多种子比较。评价单元与31类相同，但预训练来源、维度和时序分辨率不同，属于方案比较。

## 文件、资源与停止

all_seed_results.csv、summary_mean_std.csv、per_class_AP.csv、per_class_AP_stability.csv、length_and_coverage_groups.csv、coverage_warning_labels.csv、ResNet_comparison.csv为统一交付。每轮耗时与GPU/RAM峰值保存于六组独立results.json及统一表；RAM含Workers的求和可能重复计算共享页，PyTorch显存不含桌面/驱动。Seed42资源原样复用，不计入本轮新增六次训练耗时。

Defensive Rearing保持119维输出，但排除训练损失并保护专属参数行，BCE119另作诊断。固定权重Valid指标已与各results.json校验一致。全部工作仅写phase2_5，无Test、额外搜索或后续自动实验。历史保护结果见protection_verification.json。

六组训练及九组固定Checkpoint评价完成，停止等待审核。

## 数据支持的诊断结论
- mean：固定Train mAP均值96.74%，Valid 43.15%，差53.59个百分点；F1差46.13个百分点。最佳mAP轮晚于最低Valid BCE轮3/3次。波动最大的Valid类：Grooming(7个正例，AP std=12.13个百分点)、Exploring(12个正例，AP std=5.47个百分点)、Turning Around(14个正例，AP std=4.17个百分点)。
- bigru_mean：固定Train mAP均值99.92%，Valid 46.28%，差53.64个百分点；F1差48.41个百分点。最佳mAP轮晚于最低Valid BCE轮3/3次。波动最大的Valid类：Fighting(13个正例，AP std=12.23个百分点)、Spitting Venom(7个正例，AP std=8.08个百分点)、Flapping(32个正例，AP std=8.06个百分点)。
- bigru_attention：固定Train mAP均值99.69%，Valid 46.78%，差52.91个百分点；F1差46.66个百分点。最佳mAP轮晚于最低Valid BCE轮3/3次。波动最大的Valid类：Attacking(11个正例，AP std=16.00个百分点)、Being Eaten(5个正例，AP std=13.85个百分点)、Biting(12个正例，AP std=9.07个百分点)。
- 配对model2相对model1：mAP提升3/3个种子，F1提升2/3个种子；逐类提升/下降数量见head_paired_changes.csv。
- 配对model3相对model1：mAP提升3/3个种子，F1提升2/3个种子；逐类提升/下降数量见head_paired_changes.csv。
- 配对model3相对model2：mAP提升2/3个种子，F1提升1/3个种子；逐类提升/下降数量见head_paired_changes.csv。

长度分组诊断（组内AP仅覆盖有正例的主类，各组类别集合不同）：
- model1 / coverage_warning：N=9，支持11/31类，条件mAP=74.55±0.90%，固定31类F1=15.82%。
- model1 / length16：N=779，支持31/31类，条件mAP=43.75±0.61%，固定31类F1=44.80%。
- model1 / length1_3：N=33，支持18/31类，条件mAP=44.17±0.58%，固定31类F1=18.43%。
- model1 / length4_15：N=167，支持27/31类，条件mAP=41.62±0.55%，固定31类F1=32.21%。
- model1 / no_coverage_warning：N=970，支持31/31类，条件mAP=43.25±0.57%，固定31类F1=45.14%。
- model2 / coverage_warning：N=9，支持11/31类，条件mAP=73.93±3.89%，固定31类F1=17.75%。
- model2 / length16：N=779，支持31/31类，条件mAP=46.47±0.59%，固定31类F1=46.26%。
- model2 / length1_3：N=33，支持18/31类，条件mAP=42.92±4.13%，固定31类F1=13.00%。
- model2 / length4_15：N=167，支持27/31类，条件mAP=44.53±0.96%，固定31类F1=34.48%。
- model2 / no_coverage_warning：N=970，支持31/31类，条件mAP=46.45±0.42%，固定31类F1=46.52%。
- model3 / coverage_warning：N=9，支持11/31类，条件mAP=76.19±1.68%，固定31类F1=14.85%。
- model3 / length16：N=779，支持31/31类，条件mAP=47.33±1.16%，固定31类F1=46.30%。
- model3 / length1_3：N=33，支持18/31类，条件mAP=43.89±3.10%，固定31类F1=15.42%。
- model3 / length4_15：N=167，支持27/31类，条件mAP=43.13±2.53%，固定31类F1=31.73%。
- model3 / no_coverage_warning：N=970，支持31/31类，条件mAP=46.90±0.95%，固定31类F1=46.29%。

警告标签本身的召回（固定Valid阈值、只计31主类中的警告正例）：
- model1：8个警告正标签，Recall=12.50±0.00%。样本很少，只作诊断，不据此选模。
- model2：8个警告正标签，Recall=25.00±0.00%。样本很少，只作诊断，不据此选模。
- model3：8个警告正标签，Recall=25.00±25.00%。样本很少，只作诊断，不据此选模。

## 综合判断与最终核验

Attention的Valid mAP均值最高，但仅比BiGRU Mean高约0.50个百分点，且其标准差更大；BiGRU Mean的Macro-F1均值更高、跨种子更稳定。以既定mAP为主时保留Attention为主候选，同时保留BiGRU Mean；现有三种子不足以认定两者有显著差异。

三个头的固定Train−Valid mAP差均约53个百分点，支持可重复的严重泛化差距。不能单凭本轮区分标签噪声、背景/物种分布或时序信息缺失的因果贡献；没有据此启动新实验。

新增六组训练累计22.10分钟，固定权重分析94.73秒。新增训练GPU分配峰值99.18MiB，主进程RAM峰值2019.13MiB，含Workers RSS求和峰值4630.35MiB（共享页可能重复计数）。

最终复核79396个历史文件元数据、11852个受保护文件内容SHA无变化。九组固定权重评价、六组新增训练及全部分析完成；停止等待审核。
