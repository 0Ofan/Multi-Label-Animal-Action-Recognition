# AnimalAct 2.0 Phase1.5：训练前准备

本目录复用`animalact2/phase1_5_1/`已验证且未修改的Dataset、Decoder、Transform、Collate和三个Proposal模型，在其上增加统一的DataLoader、损失、训练、评价、Checkpoint及配置管理。正式数据为只读`animalact2/release_v1/`，版本`animalact2-2.0.0-release-v1`。

**当前没有正式训练。所有保存的模型状态都位于`diagnostics/`且显式标记diagnostic，不能作为正式训练初始化或正式候选。没有Test入口、没有全量RGB/特征缓存。**

## 交付入口

- `FINAL_PHASE1_5_REPORT.md`：中文验收报告。
- `FIRST_ROUND_TRAINING_PLAN.md`、`first_round_plan.json`：资源实测后的第一轮统一方案。
- `configs/model1_seed42_bce.json`、`model2_seed42_bce.json`、`model3_seed42_bce.json`：三个正式待审核配置。
- `configs/asl_reference_not_first_round.json`：可用但不自动执行的ASL配置模板。
- `data_audit/summary.json`及逐Clip/父视频CSV：全5646个Train/Valid Clip审计。
- `preparation_validation.json`、`unit_test_results.txt`：真实小批诊断、Checkpoint恢复及资源记录。
- `protection_baseline.json`、`protection_verification.json`：历史文件保护证据。
- `environment_snapshot.json`、`artifact_manifest.json`、`FINAL_VERIFICATION.json`：环境、文件校验及最后核验。

## 实现文件

| 文件 | 功能 |
| --- | --- |
| `data.py` | 复用输入接口，添加确定性逐epoch shuffle、worker种子和统一DataLoader |
| `models.py` | 只包装Phase1.5.1的Model1/2/3和官方冻结ResNet18，无新研究结构 |
| `losses.py` | BCE、ASL、Train来源的类别支持mask及零样本类别专属参数保护 |
| `schedule.py` | fixed/warmup+cosine、Valid mAP早停及状态恢复 |
| `engine.py` | 共享Train/Valid循环、按实际样本数累积梯度、选模、最终Valid阈值、日志 |
| `evaluation.py` | 复用固定31类评价，补充父视频支持数、资格标记及逐类/混淆矩阵导出 |
| `checkpoint.py` | 冻结骨干引用+完整任务头/AdamW/RNG等状态，严格配置/数据/实现指纹校验 |
| `config.py` | 配置校验、随机种子、跨模型共同协议hash |
| `train.py`、`train_model1.py`、`train_model2.py`、`train_model3.py` | 通用及分模型入口，默认仅预检 |
| `audit_data.py` | 已完成的全量Train/Valid检查；保留已有结果，不自动重跑 |
| `validate_preparation.py` | 有界诊断及小样本吞吐估计，永不调用正式Trainer.run |
| `tests/` | 损失、累积、早停、恢复、数据隔离及最终方案测试 |

## 环境

全部使用项目现有`.venv`，没有安装/升级依赖或修改系统环境变量。当前Python3.13.15、torch2.14.1+cu130、torchvision0.29.1+cu130、OpenCV5.1.0、Pillow13.0.0、NumPy2.5.4；CUDA设备是RTX5060 Laptop 8GB。官方ImageNet1K_V1权重仍引用项目内`.cache/torch/checkpoints/resnet18-f37072fd.pth`，会核对完整SHA256，不自动下载。

在项目根目录执行以下命令只预检配置，不训练：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m animalact2.phase1_5.train_model1
.\.venv\Scripts\python.exe -B -X utf8 -m animalact2.phase1_5.train_model2
.\.venv\Scripts\python.exe -B -X utf8 -m animalact2.phase1_5.train_model3
```

以下为**审核通过后才使用**的正式执行方式；本阶段未执行：

```powershell
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase1_5.train_model1 --execute-training
# Model2/3分别替换入口名称；不得同时争用本机GPU。
```

已有实验目录默认拒绝覆盖。恢复训练需要显式指定同一目录及其`last.pt`：

```powershell
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase1_5.train_model1 --execute-training --output-dir animalact2/phase1_5/runs/model1_bce_seed42 --resume animalact2/phase1_5/runs/model1_bce_seed42/last.pt
```

训练开始前要求本次全量数据审计通过；正式Trainer强制使用全部4667/979个Clip。所有运行输出限制在本目录内，不覆盖release或历史项目。

## Checkpoint与日志

每个正式实验将保存`config.json`、`provenance.json`、`history.csv/json/jsonl`、`last.pt`、按Valid mAP选择的`best.pt`、最终`results.json`、`threshold_selection.json`、`best_valid/`逐类AP和31个one-vs-rest矩阵、`valid_predictions.pt`。曲线字段完整保存，可供后续绘图；本阶段不伪造正式训练曲线。

Checkpoint不重复嵌入冻结的44MB CNN，而是保存其官方来源/完整SHA和全部任务头参数、AdamW状态、LR进度、早停、历史、随机状态、Train类别资格及数据/代码指纹。加载时从同一已验证ImageNet文件重建CNN，严格恢复任务头。更改数据、代码或配置时禁止静默resume。

```python
from animalact2.phase1_5.checkpoint import restore_model
from animalact2.phase1_5.evaluation import action_predictions

# 仅当未来正式best.pt存在时可用；当前没有正式模型。
model, policy, state = restore_model("animalact2/phase1_5/runs/model1_bce_seed42/best.pt", device="cuda")
# logits = model(batch["frames"], batch["lengths"], batch["valid_frame_mask"])
# multi_hot = action_predictions(logits, saved_valid_threshold, policy)
```

## 指标与零样本类别

正式选模只用固定31类Valid mAP。各epoch的F1曲线采用固定0.5阈值；仅在最佳checkpoint上，用完整Valid按预定网格选择最终全局阈值。没有Test选模、阈值或推理。

119维logits和原始标签保持不变。Train唯一无正例类别Defensive Rearing（ID23）不参与训练目标，119类BCE仍单独报告；另报告118个Train支持类别BCE。每步恢复该类别专属行，防止AdamW weight decay使其变化。其原始logit仍可检查，但通过`action_predictions`不会把它输出为已学会的动作；类别表明确标注unsupported。

ASL支持gamma_neg/gamma_pos/clip，当前模板为4/0/0.05，focal权重参与完整梯度，没有detach。BCE作为第一轮Proposal严格基线；本阶段的ASL小批loss数值仅证明接口可用，不证明性能优于BCE。

## 自动化检查

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m animalact2.phase1_5.tests.test_framework
.\.venv\Scripts\python.exe -B -X utf8 -m animalact2.phase1_5.tests.test_final_plan
```

前一组包含有界合成数据优化和诊断Checkpoint写入；不执行正式训练。完整工程验证还复用了Phase1.5.1的16项契约测试，总计31项。已有全量审计和真实资源测试不必重复执行；其运行器发现完成记录会拒绝覆盖。

**本阶段结束即停止等待人工审核，不自动执行正式训练、下一阶段、Test或Dashboard。**
