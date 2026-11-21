# AnimalAct 2.0 Phase1.5.1

这里是独立重写的Proposal三模型和release-v1输入链路。没有导入AnimalAct 1.0的`models/`、`features/`或`training/`，没有使用旧任务权重。默认CNN冻结、BN保持eval，仅使用已在项目内保存并验证SHA256的官方ImageNet1K_V1 ResNet-18权重。

本阶段已经完成小批功能验证；没有正式训练、没有可用于评价性能的新checkpoint、没有全量RGB或特征缓存。正式数据只读，版本为`animalact2-2.0.0-release-v1`。

## 文件

| 文件 | 职责 |
| --- | --- |
| `dataset.py` | 只从正式Train/Valid Manifest取Clip、标签和来源，惰性读取帧 |
| `video_decoder.py` | 顺序grab/retrieve准确索引，核对FPS、总帧数和实际时间戳，无近邻/重复补帧 |
| `transforms.py` | RGB、短边224、PIL双线性、完整画面、ImageNet归一化 |
| `collate.py` | 当前Batch最大长度的数值padding、lengths、valid/padding masks与来源保留 |
| `backbone.py` | 官方本地ResNet18，移除fc，只处理真实帧，默认每chunk8帧 |
| `common.py` | 长度/mask验证、masked mean、packed BiGRU |
| `model1.py` | 冻结ResNet18 + masked mean + Linear(512,119) |
| `model2.py` | 冻结ResNet18 + BiGRU(512,128×2) + masked mean + Linear(256,119) |
| `model3.py` | 冻结ResNet18 + 同结构BiGRU + 119类masked temporal attention和对应分类权重 |
| `evaluation.py` | 固定31类mAP/Macro-F1、逐类AP/TP/FP/TN/FN/P/R/F1及119类BCE，Valid阈值选择 |
| `tests/test_contracts.py` | 16项自动化数据/掩码/模型/梯度/指标契约测试 |
| `run_validation.py` | 固定每Split3条真实Clip的CPU/CUDA诊断，每模型仅一次Train参数更新 |
| `validation_results.json` | 实际样本、梯度、参数变化、时间、显存和输入一致性记录 |
| `protection.py` | 旧文件快照及历史阶段/release SHA保护 |
| `FINAL_PHASE1_5_1_REPORT.md` | 中文验收报告 |

## 环境与运行

使用项目根目录现有`.venv`：Python3.13.15、torch2.14.1+cu130、torchvision0.29.1+cu130、OpenCV5.1.0、Pillow13.0.0、NumPy2.5.4。本阶段没有安装或升级依赖；具体运行环境见验证JSON。

官方预训练文件是项目内`.cache/torch/checkpoints/resnet18-f37072fd.pth`，SHA256为`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。代码没有隐式下载行为或全局Torch缓存写入；权重缺失/不符会明确报错。

在项目根目录运行契约测试，不写测试缓存：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m animalact2.phase1_5_1.tests.test_contracts
```

需要重新运行少量真实诊断时，必须指定**新的**本阶段输出目录，防止覆盖现有验证结果；这不是正式训练入口：

```powershell
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase1_5_1.run_validation --output-dir animalact2/phase1_5_1/validation_runs/recheck_01
```

当前交付完成后已停止，不会自动执行上述复查或下一阶段。

## 统一API

```python
from torch.utils.data import DataLoader
from animalact2.phase1_5_1 import ReleaseClipDataset, collate_clips, move_batch_to_device, Model3

dataset = ReleaseClipDataset("train")  # 不在初始化时解码全部视频
loader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=collate_clips)
batch = move_batch_to_device(next(iter(loader)), "cuda")
model = Model3().to("cuda")
logits = model(batch["frames"], batch["lengths"], batch["valid_frame_mask"])
# logits: [B,119]；同一batch可原样交给Model1/Model2。
```

`valid_frame_mask=True`表示真实帧，`padding_mask=True`表示无效数值槽，二者严格互补。`lengths`为CPU int64 `[B]`，帧张量为`[B,Tmax,3,H,W]`，索引padding为-1；Tmax取当前Batch最长片段，不固定16。完整来源在`metadata`与`label_sources`，另有`clip_ids`、`parent_video_ids`、`splits`。

`forward_features(features, lengths, valid_frame_mask)`提供相同时间头接口，供自动化测试隔离验证；本阶段没有写出或读取旧特征缓存。正常原始视频路径使用`forward(frames, lengths, valid_frame_mask)`。

`return_aux=True`返回logits、512维帧特征，以及相应模型的池化/时序/注意力张量；默认只返回logits。多标签概率在外部对119个logits分别sigmoid，不能对类别做softmax。

`evaluate_multilabel`默认要求固定31类都有正例；少量功能Batch不具备完整支持时明确拒绝给出完整mAP。`require_all_main_classes=False`时不完整mAP仍为None。阈值只允许在声明为Valid且覆盖31类的输入上，从0.05到0.95、步长0.01选择Macro-F1最大值，并列取较低阈值。实际训练入口尚未实现，后续必须显式锁定Valid阈值并按父视频检查相关样本。

当前Dataset明确禁用Test；没有读取Test视频、标注或预测。不同长宽比的Clip需要后续空间分桶适配，Collate会明确拒绝不同空间形状，不能静默裁剪或拉伸。当前正式源视频统一640×360，输出398×224，不受此限制影响。
