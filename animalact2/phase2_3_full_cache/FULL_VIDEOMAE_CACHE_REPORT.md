# AnimalAct 2.0 Phase 2.3 全量VideoMAE特征缓存报告

**全量Train/Valid缓存生成、逐Clip独立核验、完整复用检查均通过。未启动训练，停止等待审核。**

## 完成数量

| Split | 成功Clip | 父视频 | 真实帧引用 | 特征时间步 | 奇数长度Clip | 失败 |
|---|---:|---:|---:|---:|---:|---:|
| train | 4667 | 2063 | 67394 | 33983 | 572 | 0 |
| valid | 979 | 442 | 14247 | 7173 | 99 | 0 |
| 合计 | 5646 | 2505 | 81641 | 41156 | 671 | 0 |

使用封存数据 `animalact2-2.0.0-release-v1`，固定Train4667、Valid979，119类标签、Clip顺序、时间范围、帧索引及标注来源不变。真实帧引用指Manifest规定的已选帧，不把长区间未选中的全部密集视频帧混入本轮。保留全部规定的真实帧，没有Test缓存、Test来源读取或Test推理。

## 与Pilot完全相同的实现

- **直接导入** `animalact2.phase2_2.adapter.FrozenVideoMAEAdapter`，没有重写或修改模型结构、编码器、空间池化或预处理。
- 既有Kinetics预训练源：`MCG-NJU/videomae-base-finetuned-kinetics`，revision `488eb9a0565f257b32866000305c8178965eb9f6`；权重SHA256 `065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73`。原权重不复制，不下载新权重，不使用AnimalAct任务权重。
- RGB完整画面、PIL bilinear短边224等比Resize、ImageNet归一化，实际224×398；无中心裁剪、拉伸、上下文扩展或重复帧。
- Patch内部只在右侧补2列归一化零值，对齐224×400；奇数L只添加一个内部归一化零值槽，原索引为null。真实像素不变，最后部分有效tubelet保留，`Lt=ceil(L/2)`，不是补到固定16帧。
- 空间位置表由14×14 bicubic插值到14×25，时间由8 linear插值到Lt，align_corners=False；原生尺寸旁路行为保持不变。全部沿用Pilot代码及SHA。
- 冻结、eval、inference_mode、CUDA **FP32**、SDPA、TF32关闭；按相同真实L分组，最大视频batch=2。没有反向传播、优化器或任务头构造。
- 输出为空间平均后的 **`[Lt,768]` FP32变长时序特征，Lt=1–8**。磁盘不Padding，不保存RGB、完整空间Token或时序分类结果。
- **Context与Pilot逐字段相等**，SHA256 `27f425cfabc59282b8db3540a24e30350f84d8340b1c6a20e179d6adc59217e9`。继续使用Pilot schema标识是为了保持缓存身份完全一致，实际全量范围由新目录的 `full_plan.json` 独立限定。

Pilot的120份有效缓存经源身份、配置、数组和SHA核验后独立写入新目录，与Pilot逐位相等；未使用软/硬链接，未改动Pilot。另外 **5526份** 为新编码缓存。

原提取进程未留下最终运行汇总，退出原因无法从持久化记录确定。恢复前逐份校验并保留5388份有效提交（Train 4667、Valid 721）；其中120份来自Pilot，5268份为原进程新生成。此次实际复用5388份，仅补齐258个Valid Clip，未重算已提交缓存。原进程在途解码次数未完整记录，不把有效提交数冒充精确累计解码次数。

## 缓存身份及断点续跑

缓存键直接复用Pilot `cache_context()`、`descriptor()`和SHA256算法，包含预训练源/代码/预处理、视频内容SHA、Clip/父视频/Split、真实索引、有效时间及标签来源摘要。

存储仍为每Clip一个目录：`features.npy` + `receipt.json`。NPY为未压缩FP32，收据保存内容SHA、形状、119维target、完整label_evidence、tubelet到真实帧/null的映射、真实帧计数及有效Mask。先在`.pending_*`目录写入、flush/fsync，再原子重命名提交；有效提交仅核验复用，拒绝覆盖；损坏提交明确报错，不静默重算覆盖。机制为单写入器，不支持并行多写入进程。

全量完成后已实际运行一次完整复用检查：**复用5646，新增提取0、复制0、视频解码0，编码器未构造**，耗时 **15.44秒**。没有为了验证续跑重复GPU推理。

## 独立全量核验

验证入口不调用生成函数、不构造模型，不解码视频；逐一读取已保存NPY、JSON收据及正式Manifest：

1. **5646/5646** 数组均为连续FP32、`[ceil(L/2),768]`、无NaN/Inf；文件SHA匹配，唯一缓存键无重复、无缺失、无额外Clip、无未提交目录。
2. 帧到tubelet的映射展开后与原真实帧索引逐项相等，无重复补帧或遗漏；671个部分有效末tubelet仍有效。所有Lt位置至少含一个真实帧，真实帧计数合计81641。
3. 原119维标签及完整label_evidence一致，Clip、父视频、Split、精确有效时间和原始annotation引用对齐。核对 **10638** 个正Clip×类别组合、**11732** 条贡献标注关系。
4. 在生成前及最终核验中，2505个Train/Valid源视频和2505个原JSON的内容SHA与release匹配，共 **5010个源文件**；不重新设计或修改标签规则。
5. Pilot对应120份特征及收据与全量缓存一致，数组逐位相等且文件SHA相同。
6. 两Split真实缓存各取原长度1、3、16的样本，得到特征长度1、2、8，Batch为 `[3,8,768]`、target `[3,119]`，Padding Mask及零值填充正确。仅检查数据接口，模型前向、反向和参数更新均为0。
7. 输出119类及固定31主类支持数表。Defensive Rearing（ID23）训练正例仍为0，未改类别定义；本阶段不进行性能评价或阈值选择。

独立核验耗时 **12.95秒**，失败0。证据见 `clip_validation.csv`、`validation_summary.json`、`split_statistics.csv`、`class_support_119.csv`、`main31_support.csv`、`source_checks.csv`、`reader_interface_checks.json` 和空的 `errors.json`。

## 实际资源

- 已有日志记录的累计构建耗时**至少1329.22秒（22.15分钟）**：中断前最后持久化进度1215.86秒，加本次完整续跑113.36秒。该值是下限；末次进度后提交及退出前未记录的时间无法精确恢复，中断等待、恢复预检及最终核验另计。 时间包含各自运行中的源身份/缓存核验、模型加载、编码和落盘，不是纯GPU Kernel时间。
- 资源范围：**本次补齐258个Clip的续跑段**。中断前进程的最终GPU/RAM峰值未落盘，无法追溯；下列实测值只覆盖本次续跑，不能当作整个生成过程峰值。
- 此范围内PyTorch显存分配峰值：**648.62MiB**，保留峰值 **772.00MiB**；硬件为RTX5060 Laptop 8GB。
- 此范围内进程RSS峰值：**1.588GiB**，20ms采样；无DataLoader子进程，不当作精确独占物理RAM。
- 纯特征数据：**120.57MiB**；包含NPY头的数组共 **121.26MiB**。
- 完整缓存含收据/context：**158,034,346字节，150.71MiB（0.147GiB）**，11293个文件。
- Windows实际文件分配空间：**194,052,096字节，185.06MiB**，读取FILE_STANDARD_INFO.AllocationSize后逐文件求和；不包括目录索引/MFT开销，不等同整个分区空间变化。
- Train/Valid缓存逻辑大小分别为 **124.47 / 26.25MiB**。未重复保存预训练权重。
- 生成失败0、最终核验失败0。全部阶段新增文件清单/SHA和大小另见 `artifact_manifest.json`、`FINAL_VERIFICATION.json`；其中运行时临时目录不纳入交付清单。

## 后续读取接口

```python
from torch.utils.data import DataLoader
from animalact2.phase2_3_full_cache.dataset import CachedFeatureDataset, collate_features

dataset = CachedFeatureDataset("train")  # 或valid，Test入口禁用
batch = next(iter(DataLoader(dataset, batch_size=8, collate_fn=collate_features)))
# batch["features"]: [B,Lt_max,768] FP32
# batch["lengths"]: 真实tubelet长度，CPU int64
# batch["valid_frame_mask"] / batch["padding_mask"]: 仅临时Batch Padding
# batch["target"]: [B,119]；Clip/父视频ID、真实帧及来源信息仍保留
```

新Dataset与Pilot的三个768维分类头接口一致，但**本阶段没有运行分类头或训练器**。VideoMAE特征依赖整个Clip上下文，不能作为与上下文无关的独立逐帧特征复用。Pilot已披露的单帧运动证据不足、部分有效tubelet及位置插值分布问题仍然存在；缓存正确不代表已证明分类性能提升。

## 保护与停止

已有 **67,987** 个文件元数据及 **444** 个受保护文件内容SHA复核不变，包含正式Train/Valid、完整VideoMAE Pilot、预训练权重及ResNet18第一轮结果。原视频/JSON还逐一核验内容SHA。大型其它历史文件只检查元数据，不能声称全部历史内容均重新哈希。

新代码及所有产物仅写入 `animalact2/phase2_3_full_cache/`。原release-v1、ResNet18历史结果、Pilot、旧缓存、模型和配置保持原状；网络及Test读取受运行期保护阻断，没有下载新权重、读取Test视频/标注/Manifest/预测、提取Test特征、反向传播、训练、评估或开发前端。

**Phase2.3完成，停止并等待审核，不自动进入正式训练。**
