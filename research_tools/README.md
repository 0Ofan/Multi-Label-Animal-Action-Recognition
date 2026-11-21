# AnimalAct 2.0 完整研究复现与继续开发

本包补齐网页之外的研究资料。只包含当前2.0；不包含AnimalAct1.0实际代码、权重、缓存、预测或旧模型结果。原始AnimalAct视频、原JSON标注及demo仍由课程成员本地保存，不公开上传。本次按用户授权公开派生Clip标签、来源索引及缓存收据。

## 文件在哪里

- Git主仓库：当前Dashboard、模型/训练代码、配置、汇总结果、选取的数据质量基础资料、本工具及课程Brief/Proposal。
- Release `v2.0-research-reproduction`：完整研究快照，约995MiB压缩、43,903个文件，解压约1.48GiB。每个文件有SHA256，包含原始封存版本的源代码、数据清单和成果。
- release_v1：Train4667、Valid979、Test1022个正式Clip；119类定义、31主类、帧索引、有效长度、label_evidence和标注UID均保留。
- 两套全量FP32缓存：ResNet18 [L,512]、VideoMAE [ceil(L/2),768]；含帧/Clip索引、来源、预处理、权重指纹及SHA。
- 模型：ResNet18第一轮3组、VideoMAE Seed42三组、Seed123/2026六组，共12组正式结果与24份Best/Last Checkpoint；另有原有诊断记录。
- 原保存的Train/Valid预测、阈值搜索、逐类AP、混淆矩阵、历史曲线、资源记录和数据质量报告。
- 原ImageNet ResNet18与Kinetics VideoMAE预训练文件，便于不依赖本机旧缓存重建实验。
- 历史完整性基线可能列出旧项目的路径/哈希，这是2.0封存审计的一部分，绝不等于包含这些1.0文件内容。Phase0中真正依赖旧模型的审计代码/结果被排除；仅选取原始数据质量部分作为补充。

## 1. 环境

Python3.13。可用仓库根requirements.txt建立CPU环境；有合适NVIDIA显卡时，在独立训练环境安装：

```powershell
python -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install -r research_tools/requirements-cuda.txt
```

CUDA依赖对应原cu130环境，需支持的显卡与驱动；不会自动修改系统环境变量。Linux使用相应的venv/bin/python。完整原Windows训练环境的版本记录见environment-windows-training.lock.txt；它是历史环境证据，不应盲目安装其中所有Windows专用依赖到Linux。

## 2. 恢复研究包

```powershell
python research_tools/restore.py
```

默认恢复到仓库下 `research_workspace/`，不覆盖网站代码。可指定 `--target D:\AnimalAct2Research`。下载、解压及每个文件都校验SHA；已有文件相同则跳过，不同则拒绝覆盖。

不要把包解压到正在运行的网站animalact2目录。网页使用经验证的便携版模块；研究快照保留历史封存代码，二者隔离，避免改变旧Checkpoint的源代码指纹。

## 3. 核验同学本地课程数据（推荐）

```powershell
python research_tools/check_dataset.py --workspace research_workspace --data-root D:\Datasets\AnimalAct
```

只对正式Train/Valid所涉及的源视频/JSON做内容SHA核验，不读Test。文件复制后的mtime可以不同，只要内容一致。缓存训练本身可不读取原始视频；视频级改进、重新提取特征与未来定位任务需要原始数据。

## 4. 先做诊断，不自动正式训练

```powershell
python research_tools/run.py --workspace research_workspace --family videomae --model model3 --smoke
python research_tools/run.py --workspace research_workspace --family resnet18 --model model1 --smoke
```

每次仅8个真实Train Clip、32个Valid Clip，执行1次诊断性优化器更新，不报告小样本31类mAP，不保存正式模型。model1/model2/model3分别对应Mean、BiGRU Mean、BiGRU Class-specific Temporal Attention。默认自动选择CUDA/CPU。

## 5. 复算已有Checkpoint的完整Valid结果

```powershell
python research_tools/run.py --workspace research_workspace --family videomae --model model3 --evaluate research_workspace/animalact2/phase2_4/runs/model3_bce_seed42/best.pt --output research_workspace/validation_model3.json
```

完整979个Valid，优先使用原Checkpoint目录的Valid阈值；不根据Test选择任何参数。其他种子使用对应phase2_5/runs目录。ResNet18使用family=resnet18，Checkpoint来自first_round/runs。

读取原Checkpoint可重新评价；继续研究建议以新实验初始化，保留原结果。若要精确续跑原历史训练器，其Windows/CUDA环境、封存代码和校验规则仍然适用；不要为了通过校验去改写原Manifest或伪造文件时间。

## 6. 明确启动新的训练/改进实验

```powershell
# 原协议：Seed42、BCE118、Batch8×累积4、AdamW、LR1e-3、WD1e-4、2轮warmup+cosine、50轮、Patience10
python research_tools/run.py --workspace research_workspace --family videomae --model model3 --train --seed 42
# 需要时自行运行ASL对照；本次上传工作没有执行这个实验
python research_tools/run.py --workspace research_workspace --family videomae --model model3 --train --loss asl --gamma-neg 4 --seed 123
```

新结果只允许写入research_workspace/new_experiments/，默认唯一目录，拒绝覆盖旧结果。可指定 `--output research_workspace/new_experiments/my_run`，后续用同样参数和 `--resume .../last.pt` 恢复。新入口使用原2.0损失、Epoch训练数学、调度、Sampler和评价函数，不修改模型结构。Defensive Rearing仍保留119维输出但排除损失与正常发出。

按Valid固定31类mAP选Best，最终用Valid Macro-F1选择阈值；保存Best/Last、优化器、RNG、配置、源代码/数据/缓存指纹、早停、历史及Valid结果。Train online曲线不混称固定Checkpoint Train分数。跨硬件重新训练可能有浮点差异，不承诺每个小数位相同。

旧封存入口针对固定实验，因此可能直接跳过已完成运行，或检查旧路径、mtime和全目录历史基线。它们作为源代码证据保留；异机复现和新实验使用上述portable入口，避免更改封存文件。缓存按来源、标签、预处理和文件内容SHA核验，不以mtime相同冒充数据相同。

## 7. 重新使用冻结VideoMAE及继续开发

`research_tools/common.py` 的load_encoder(root,device)会把绝对路径解析到恢复包中的实际预训练文件，仅创建运行时指针，不改原代码/权重/缓存描述。复用Phase2.2完整画面、Patch对齐、变长/奇数帧和位置编码逻辑。

微调视觉Backbone、改采样或添加空间增强需要在线帧训练/新缓存，不能拿旧冻结特征假装完成微调。可在新实验目录基于现有Dataset/decoder/transform/encoder开发，保留固定Split和标签映射。

Bonus若为Temporal Video Grounding：本包已有原标注时间来源、帧映射、Clip数据与Backbone基础，但定位头、时间段预测、定位损失和专用评价仍需按正式Assignment Brief另行实现。分类Attention或用户输入窗口不是定位预测。**未声称已完成加分项，也未在本轮开展它。**

## 保护与验证范围

Test Manifest仅隔离保存；现有缓存/模型选择/新训练入口只使用Train/Valid。本轮不重新训练正式模型，不重提取全量特征，不修改原数据与历史实验，不上传原始视频，也不上传虚拟环境或凭据。网页模型、阈值、CPU部署入口保持不变；研究包不会被网站自动下载。

## 本次实际验收

- 从独立目录恢复43,903个原封存文件，逐文件SHA256核验通过。
- 六种Backbone/分类头组合（ResNet18与VideoMAE，各三个头）的真实Train/Valid Batch、Mask、梯度及一次诊断更新通过。
- 12份正式Best Checkpoint在完整979个Valid Clip上复算；mAP、Macro-F1、BCE118、BCE119最大差异均为0，119类TP/FP/TN/FN及正例计数一致。
- 校验5,010个Train/Valid源视频/原标注文件内容SHA，全部匹配；这些原始文件没有上传。
- 恢复后的VideoMAE权重实际加载、路径重定位、单帧/矩形对齐通过，输出[1,1,768]且完全冻结。
- 没有新增正式训练、全量特征重提取或Test推理；新入口50轮正式训练及新实验效果不属于本轮已完成实验。

结果汇总：deployment/research_validation.json。新实验不自动替换网页模型；必须先按固定Valid协议审核，再明确导出部署。网页仍只做推理展示，训练在成员自己的计算机或自有服务器执行。
