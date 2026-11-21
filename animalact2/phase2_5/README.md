# Phase 2.5 多种子验证及过拟合诊断

复用Phase2.4的三个Seed42正式结果，仅训练Seed123和2026的六组。分类头、缓存、118类BCE、优化器、调度、Batch、早停及Valid选模规则全部固定；模型工厂只接受42/123/2026，配置与Phase2.4除Seed外必须一致。

从项目根目录使用项目`.venv`：

```powershell
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_5.run
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_5.analyze
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_5.support verify
```

`run`在已保存last.pt后中断可用同一命令恢复，完成结果跳过。诊断权重不能用于正式训练初始化。`analyze`只加载九个最佳头，分别计算完整Train和Valid的固定Checkpoint指标，不训练、不调整阈值；固定预测保存于本目录，可在分析中断后校验Checkpoint SHA后复用。完成的分析拒绝重写。

主要交付：

- `all_seed_results.csv`：九组固定Train/Valid指标、阈值、泛化差距、最佳mAP轮及最低BCE118/BCE119轮、资源；Seed42原训练资料只读引用。
- `summary_mean_std.csv`：每个头N=3的均值及样本标准差ddof=1。
- `per_class_AP.csv`、`per_class_AP_stability.csv`：31主类逐seed及跨seedAP与支持数。
- `head_paired_changes.csv`：相同seed不同头的mAP/F1变化及逐类AP提升/下降数量。
- `length_and_coverage_groups.csv`、`group_mean_std.csv`：真实帧长度1–3、4–15、16及覆盖警告分组。缺失主类正例时完整31类mAP留空，supported_main_mAP仅为支持类条件均值，不能当成完整31类主指标。
- `coverage_warning_labels.csv`：仅31主类警告正标签的Recall，不对全正例标签集合计算AP/Precision。
- `ResNet_comparison.csv`：ResNet单Seed42与VideoMAE三种子汇总的Valid对照，重复次数不对等。
- 六组独立`runs/`目录：权重、历史、配置、资源及曲线；所有新结果均在本阶段。

Train online曲线与固定Checkpoint Train指标分别保存。后者与Valid使用同一权重、同一Valid阈值。训练主目标BCE118，BCE119为额外诊断；Defensive Rearing保留输出及既有排除规则。

不读取Test，不重新提取特征、不微调、不增加增强或搜索，不修改历史实验。全部训练和分析完成后停止等待审核。
