# 共享ResNet18特征缓存 Pilot

仅限`selection.json`锁定的100 Train、20 Valid Clip。没有全量特征，没有Test，没有正式训练，也没有修改既有模型、训练器或正式配置。

## 文件

- `selection.json`、`selected_manifest.jsonl`：固定样本、原始来源、数据/处理指纹。按长度分层并加入共享帧对，不按模型效果挑样本。
- `cache/context.json`：完整画面短边224、RGB/ImageNet归一化、FP32、官方ImageNet ResNet18权重SHA及实现版本。
- `cache/frames/shard_*/features.npy`：最多8个唯一帧的`[N,512]`FP32输出。
- `cache/clips/{train,valid}/{clip_id}/features.npy`：每Clip的`[T,512]`FP32序列，T为原始真实长度，不保存padding。
- 每个数组同目录`receipt.json`：来源/缓存键、形状、dtype、文件字节数和SHA256。数组内只有ResNet输出，不包含标签、BiGRU或Attention结果。
- `extraction_runs/`：主动中断、续跑、完整复用三次记录。
- `validation_results.json`：120条逐模型logits比较、梯度及实时/缓存单轮对照。
- `fresh_process_cached_resources.json`：独立新进程的缓存Model3内存/显存复测。
- `CACHE_PILOT_REPORT.md`：完整中文结论及全量Train/Valid容量、耗时估算。

帧键为源视频ID/Split/内容SHA、真实帧索引、预处理指纹及Backbone权重SHA的组合哈希；context还记录Torch/Torchvision/Pillow、处理代码、FP32及TF32设置。Clip键另外绑定帧键序列、Clip ID、119维target及原标注来源摘要。读取时验证当前Manifest、源视频stat、完整描述、数组SHA/形状/dtype/有限值。标签和来源从只读release-v1取得，不能由特征文件生成或替换。

## 续跑与完整性

每个帧分片/Clip先在`.pending_*`目录写完整FP32数组和receipt，再原子改名提交。已提交且通过验证的条目只读复用；未提交目录不作为缓存。遇到已提交条目损坏或上下文不符，明确报错，不静默复用/覆盖。中断时最多重做尚未提交的计算，不会重提取已确认有效的帧。

本轮实际在68个已提交帧处主动暂停，续跑只新增1631帧，合计1699个唯一帧；第三次运行新增CNN帧数为0。这里的“唯一帧只提取一次”指缓存构建；为做实时对照而重新运行CNN的计算单独计时，不修改缓存。

在项目根目录使用现有`.venv`。以下命令始终只处理同一120条，不提供`--all`或Test选项：

```powershell
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase1_5_cache_pilot.extract
```

可选`--pause-after-new-frames 64`在完整分片提交后暂停，最多越过阈值7帧。已完成的缓存再次执行时只检查/复用，并另存运行记录。本次交付后已停止，没有自动执行。

## 共享模型接口

`CachedFeatureDataset`产生真实`[T,512]`，`collate_features`仅按Batch最长序列建立数值padding和Mask。为复用现有训练循环，batch里的`frames`槽承载特征并标注`input_kind=frozen_resnet18_features`。

```python
from animalact2.phase1_5_cache_pilot.dataset import CachedFeatureDataset, collate_features, FeatureForwardView
from torch.utils.data import DataLoader

dataset = CachedFeatureDataset("train")
batch = next(iter(DataLoader(dataset, batch_size=8, collate_fn=collate_features)))
# 原model仍是Phase1.5.1的Model1/2/3，只有调用入口不同：
# logits = model.forward_features(batch["frames"], batch["lengths"], batch["valid_frame_mask"])
# view = FeatureForwardView(model)  # 无层、无参数，只调用原模型forward_features
```

三个模型读取完全相同缓存。模型对象和参数定义不变，CNN冻结/eval；BiGRU、Attention及分类参数仍由原优化器正常更新。正式配置仍是原文件：batch8、累积4、BCE、AdamW、seed42及原类别支持策略，均未改写。

`validate_pilot.py`只跑该100/20子集各一轮用于时间/梯度诊断，拒绝覆盖完成记录；不调用正式`Trainer.run`，不选最佳模型、不调阈值、不保存时序输出或任务权重。`resource_probe.py`另在新进程只测缓存Model3的一轮资源。它们都不是正式训练入口。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m animalact2.phase1_5_cache_pilot.tests
```

测试中的损坏/未提交样例只在`test_fixtures/`，真实`cache/`完整。该方案只适合完全冻结CNN、固定帧和固定预处理；若改变帧、增强、归一化或Backbone权重，必须重新生成匹配键的缓存。

**全量估算仅指Train4667+Valid979；未读取Test清单、视频、标注或预测。完成后等待审核，不自动生成全量缓存。**
