# AnimalAct 2.0 完整Train/Valid ResNet18共享缓存报告

**完整缓存生成和独立核验通过。未训练、未修改Checkpoint，现停止等待审核。**

## 1 实际完成数量

| Split | 成功Clip | 父视频 | 唯一帧 | Clip帧引用 | 失败Clip |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 4667 | 2063 | 60744 | 67394 | 0 |
| valid | 979 | 442 | 13140 | 14247 | 0 |
| 合计 | 5646 | 2505 | 73884 | 81641 | 0 |

Train4667、Valid979，与release-v1正式顺序及标签一致。没有Test缓存或Test来源读取。每个唯一帧输出512维FP32；Clip保存真实`[T,512]`，T=1–16，不在磁盘上Padding、重复补帧或保存时序输出。

## 2 与Pilot完全相同的条件

复用Pilot的context、帧键/Clip键、label元数据摘要、数组读取和SHA校验函数；Full写入函数仅将输出边界改到新目录，沿用相同NPY、收据、`.pending_*`和原子目录提交格式。

- 官方权重：ImageNet ResNet18 IMAGENET1K_V1，文件SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。
- context SHA256：`d3b735f733ffda4e28b5c7da9c14a299d24df4b6efd35e2b1b9b5e8c41aac4a4`，与Pilot逐字段相同，包含代码/库版本。
- 完整画面、RGB、PIL bilinear、短边224等比Resize，当前398×224，ImageNet归一化，无中心裁剪/letterbox。
- 使用release-v1已确认的真实帧索引和变长规则；CNN冻结、eval、inference_mode，chunk8、FP32、TF32关闭。

先核对2505个源视频和对应原Train/Valid JSON的内容SHA。Pilot1699帧经SHA、身份键和上下文校验后独立复制到Full目录，不用软/硬链接、不改Pilot；只为另外**72185帧**运行CNN。共73884个唯一帧各有一份有效输出，没有重复提取已确认有效的帧。

## 3 断点续跑与完整性

每个分片/Clip由临时目录完整写数组、fsync及SHA收据，再原子提交。已有提交必须通过SHA、身份描述、形状、dtype和有限值检查才复用；未提交临时目录不当有效缓存，损坏提交明确报错，禁止静默覆盖。沿用单写入器机制，不提供并发写锁。

完成后已实际再次执行完整流程：新增CNN帧数**0**、视频解码**0**、新增Clip**0**，复用73884帧和5646个Clip；耗时**27.66秒**。不为验证续跑重新计算已有特征。

独立核验逐个检查：

1. 全部帧库数组FP32、512维、无NaN/Inf、SHA/字节数与收据一致，73884个唯一键完整，无额外或重复键。
2. 全部5646个Clip形状等于真实长度，原帧顺序、Clip/视频/Split映射正确，每一行与共享帧库**逐位一致**。
3. 重新读取原Train/Valid JSON，按批准的终点截断及半开区间求标签交集，核对119维target、密集/采样支持、原Clip窗口、annotation UID/Pointer、动物名称及贡献交集。
4. 实际核验10638个正Clip×类别组合、11732条标签来源关系；31主类不变，Defensive Rearing训练正例仍为0。
5. Pilot对应120个Clip与Full缓存逐位相等，全部历史Pilot文件保持原状。

独立核验约**93.41秒**，无额外CNN推理。详细结果见`validation_summary.json`、`validation/clip_checks.csv`、`source_checks.csv`、`split_statistics.csv`、119类/31主类支持表及空的`errors.json`。

## 4 实际资源

- 构建总耗时：**683.27秒，约11.39分钟**，含源身份校验、Pilot复制、新帧提取和Clip组装；不含后续独立核验、完整复用检查及报告整理。
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，总显存7.96GiB。
- 提取PyTorch分配峰值：**194.85MiB**，保留峰值256MiB。不含驱动、桌面等其它显存。
- 提取进程RSS峰值：**1.56GiB**，Windows工作集20ms采样，不当作精确独占物理RAM。
- 缓存逻辑文件大小：**369,019,726字节，351.92MiB（0.344GiB）**，共30585个文件。
- Windows文件分配空间：**477,278,208字节，455.17MiB**。由FILE_STANDARD_INFO.AllocationSize逐文件读取，包含文件分配余量；不含目录索引/MFT等文件系统元数据，不能把它称为整个分区变化量。
- 其中帧库逻辑大小182.09MiB，Clip数组及收据169.83MiB；既有官方权重不重复复制。
- 生成失败0，最终核验失败0。

## 5 三模型读取接口

统一入口在本目录`dataset.py`：

```python
import torch
from torch.utils.data import DataLoader
from animalact2.phase1_5_cache_full.dataset import CachedFeatureDataset, collate_features
from animalact2.phase1_5_1.collate import move_batch_to_device

dataset = CachedFeatureDataset("train")  # Valid改为"valid"，Test不可用
loader = DataLoader(dataset, batch_size=8, num_workers=0, collate_fn=collate_features)
batch = move_batch_to_device(next(iter(loader)), "cuda")
with torch.no_grad():
    logits = model.forward_features(batch["frames"], batch["lengths"], batch["valid_frame_mask"])
```

model为原Model1、Model2或Model3；三者使用同一个Dataset和相同缓存。batch通用输入槽`frames`是`[B,Tmax,512]`，不是RGB；标签`[B,119]`。只在临时Batch做动态数值padding，BiGRU沿用真实lengths，Mean/Attention沿用mask。Clip ID、父视频、Split及完整来源仍可读取。

`FeatureForwardView(model)`复用Pilot无参数适配器，`make_feature_loader`按现有正式配置构建loader；没有新模型层，没有改训练参数。现有RGB训练入口未自动切换；后续正式采用缓存时需显式使用该接口，不把特征直接交给RGB forward。

5项接口测试通过；三个模型各读取同一个含Train/Valid长度1/10/16的6样本Batch，输出`[6,119]`且有限，mask与原数据一致。Model3无效时间Attention为0。检查期间CNN调用0、反向0、参数更新0，没有训练或写Checkpoint。

## 6 文件保护与停止

仅在`animalact2/phase1_5_cache_full/`写新文件。核对37021个旧文件大小/修改时间、1352个历史文件SHA（包含Pilot所有缓存）、官方ImageNet权重SHA，均未变。原始数据、release、模型、训练配置和Checkpoint不修改；Test视频、标注、Manifest及预测不打开，不下载依赖/模型。

最终全目录文件数/大小见`FINAL_VERIFICATION.json`，发布文件SHA清单见`artifact_manifest.json`。**完整缓存交付后停止等待审核，不启动正式训练、不修改Checkpoint、不自动进入下一阶段。**
