# 完整Train/Valid ResNet18共享缓存

只包含release-v1的Train4667与Valid979个Clip，73884个唯一帧。没有Test、正式训练、任务Checkpoint或时序特征。本目录独立保存，不链接或覆盖Pilot。

## 格式与来源

- `full_plan.json`：固定Train/Valid Clip ID、源视频/标注身份、源stat、Manifest SHA及完整处理context。
- `cache/context.json`：与Pilot逐字段相同的ImageNet ResNet18权重、完整画面短边224、RGB、归一化、FP32及版本指纹。
- `cache/frames/shard_*/features.npy`：每分片`[N,512]`，N≤8，唯一帧库。
- `cache/clips/{train,valid}/{clip_id}/features.npy`：每Clip真实`[T,512]`，1≤T≤16，不保存padding。
- 每个数组的`receipt.json`：沿用Pilot的缓存键、形状、dtype、字节数、SHA256和来源摘要。FP32 `.npy`，不压缩、不量化，不存BiGRU/Attention结果。
- `extraction_runs/`、`validation/`、`validation_summary.json`、`reader_interface_checks.json`：运行及逐Clip/逐类核验依据。
- `FULL_CACHE_REPORT.md`、`FINAL_VERIFICATION.json`、`artifact_manifest.json`：中文报告、统计及文件完整性清单。

原始119维标签、帧索引和annotation来源从只读release-v1取得；缓存描述把这些元数据的摘要绑定到对应数组。帧键包含视频ID/Split/内容SHA、帧索引、预处理及权重哈希。相同帧只运行一次CNN，多个Clip共享其结果。

Pilot的1699个有效唯一帧经完整SHA和键校验后独立复制，剩余72185帧新提取。Full和Pilot的相同120个Clip应逐位相等。复制不使用软/硬链接，不改变Pilot文件。

## 三个模型读取同一缓存

在项目根目录使用现有`.venv`。不需要安装新依赖或下载权重。

```python
import torch
from torch.utils.data import DataLoader
from animalact2.phase1_5_cache_full.dataset import CachedFeatureDataset, collate_features
from animalact2.phase1_5_1.collate import move_batch_to_device
from animalact2.phase1_5.config import load_config
from animalact2.phase1_5.models import build_model

dataset = CachedFeatureDataset("train")  # Valid用"valid"，Test明确禁用
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_features)
batch = move_batch_to_device(next(iter(loader)), "cuda")

config = load_config("animalact2/phase1_5/configs/model1_seed42_bce.json")
model = build_model(config).to("cuda").eval()
# Model2/3只替换对应配置，缓存和batch完全不变。
with torch.no_grad():
    logits = model.forward_features(batch["frames"], batch["lengths"], batch["valid_frame_mask"])
```

`batch["frames"]`是现有训练循环的通用输入槽，此处为`[B,Tmax,512]`并标注`input_kind=frozen_resnet18_features`；标签为`[B,119]`。`lengths`为CPU int64，`valid_frame_mask=True`表示真实帧，`padding_mask`与其相反。只在临时Batch内做数值padding，不改变磁盘Clip长度；padding帧索引=-1。

`FeatureForwardView(model)`复用Pilot的无参数调用适配器，直接调用原模型`forward_features`，可与既有共用epoch函数衔接；`make_feature_loader(dataset,config,training=...)`使用原batch/worker/seed配置构造loader。它们没有新层，没有改模型定义或任何正式配置文件。

本轮没有改写`phase1_5.Trainer`的原始RGB入口，也没有自动将其切换为缓存或开始训练。正式训练采用缓存时，应在后续明确授权的入口中使用上述数据/调用接口，而不是把512维特征误传给原RGB `model.forward`。

## 续跑与校验

```powershell
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase1_5_cache_full.extract
```

此命令只作用于固定Train/Valid；已提交且SHA/键/形状检查通过的数组只读复用。数组和收据先写`.pending_*`目录，再原子提交；未提交项不当作有效缓存。损坏的已提交项明确报错，不静默覆盖。当前是单写入器，不提供并发写锁。

`--pause-after-new-frames N`可在完整分片边界暂停（最多超出阈值7帧）；再次运行继续缺失帧。本轮完成后会执行一次完整复用检查，确认新增CNN计算0。

```powershell
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase1_5_cache_full.validate
```

独立验证重新核对原Train/Valid标注、帧点、119位标签、来源、全部SHA及Clip/帧库逐行一致性，不重新运行CNN。读取接口默认要求完整核验通过，每个Clip读取仍核对收据、SHA、形状、dtype、Manifest及源stat。该缓存只适用于当前冻结CNN、固定采样和预处理；更改这些条件必须使用匹配的新键。

**任务完成后停止等待审核。不读取Test，不自动训练或进入下一阶段。**
