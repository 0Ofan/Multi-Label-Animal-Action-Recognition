# AnimalAct 2.0 Phase 2.4 VideoMAE第一轮正式训练

固定release-v1 Train4667、Valid979，直接读取Phase2.3变长 `[T,768]` FP32缓存，T=ceil(原真实帧长度/2)。详细结果见 `FINAL_PHASE2_4_REPORT.md`，统一表见 `comparison.csv`。

## 三个头与协议

直接复用Phase2.2的 `VideoMAEProposalHead`：Mean、单层双向GRU(hidden128)+Mean、同BiGRU+类别特异Temporal Attention。119维独立Logits，变长Packed Sequence和Mask，无额外投影、Dropout或新结构。

Seed42、BCE118（保留119维输出，Defensive Rearing ID23沿用原规则）、Batch8×累积4、AdamW、LR1e-3、WD1e-4、Grad Clip5、2 Epoch Warmup+Cosine到峰值10%、最多50轮、Patience10。Valid固定31类mAP选模，最佳模型仅在Valid网格选择F1阈值。三个头独立初始化。

复用原共享epoch/损失/评价/调度代码，不修改ResNet18历史实现。缓存训练模型只包含可训练头及一个无参数的冻结Backbone来源引用，**不构造或执行VideoMAE编码器，也不读取预训练权重Tensor**。Checkpoint包含全部可学习参数、AdamW、调度、早停、RNG、历史及来源指纹；完整预训练源另在既有项目目录保留。

## 输入接口

`data.CachedFeatureDataset`读取并校验数组、收据、标签及原始Manifest。特征身份使用release内已封存的视频SHA，避免在每次取样时重新打开源视频。训练运行期保护拒绝任何原始视频/标注、Test、旧预测、旧任务权重和编码器权重加载，只允许读取正式VideoMAE数组。

`data.collate_features`沿用Pilot有效tubelet长度及Mask；共享训练器通用槽 `batch["frames"]` 在本入口表示 `[B,Tmax,768]` 特征，并以 `input_kind=frozen_videomae_temporal_features`标识。CPU int64 `lengths`、有效/无效Mask、target119、Clip ID、父视频ID、真实原长度均保留。数据项 `metadata` 保留原始帧及完整标签来源。

## 运行入口

从项目根目录使用既有 `.venv`，无需安装或下载：

```powershell
# 历史保护基线（已有则保留），只读哈希旧权重，不反序列化
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_4.support

# 锁定配置及完整缓存SHA预检
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_4.preflight

# 小量Train更新、完整Valid评价、Mask及Checkpoint精确恢复；正式训练门禁
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_4.validate_integration

# 依次三个头；中断后用同命令从last.pt恢复，已完成实验跳过
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_4.run_all --execute-training

# 三模型完成后历史/缓存保护复核
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_4.support verify

# 保存输出复核、少量Valid缓存恢复测试、曲线、比较表及中文报告
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_4.finalize
```

诊断只产生独立diagnostics权重，不能恢复到正式运行；正式入口重新初始化。Completed结果不覆盖。运行时临时文件及进程内TEMP/HF默认缓存均在本目录，系统环境变量和全局Python不变。

## 保存结果

每个 `runs/modelN_bce_seed42/` 包含Best/Last Checkpoint、精确配置/来源、history CSV/JSON/JSONL、训练曲线SVG/PNG、results.json、Valid Logits和target/Clip ID、Valid阈值搜索，以及119类/31主类AP和混淆统计。

`comparison.csv`为三头比较，`per_class_comparison_main31.csv`为逐类分析；`backbone_comparison.csv`、`backbone_paired_changes.csv`、`backbone_per_class_main31.csv`只使用已有ResNet18 Valid资料建立对应对照。`artifact_manifest.json`保存交付SHA，排除自身、最终验证文件和运行临时目录；最终验证文件单独保存清单SHA，避免自引用。

Train指标为Epoch内更新前的online混合状态，Valid为固定Epoch结束模型。历史F1阈值0.5，最终F1为最佳Checkpoint的Valid所选阈值。BCE118是优化目标；BCE119包含不可学习类，不能混称为训练Loss。Backbone比较还涉及预训练来源、输入维数及时间分辨率差异，不是纯网络结构因果试验。

完成后停止等待审核：不读Test、不重提特征、不做ASL/微调/额外消融、不开发前端或自动进入下一轮。
