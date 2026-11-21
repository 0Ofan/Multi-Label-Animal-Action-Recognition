# AnimalAct 2.0 Phase1.5.1 三个Proposal模型独立实现报告

**实现与小批CPU/CUDA验证完成。已停止等待审核，未进入Phase1.5.2或正式训练。**

## 1 实现范围与数据依据

全部新代码、测试及验证记录独立保存在本目录，没有导入AnimalAct1.0旧模型、分类头、训练器或特征模块，也没有使用旧任务checkpoint。阅读Phase0至1.4.3的报告、Proposal原文及正式数据说明后，以已封存的`animalact2-2.0.0-release-v1`为唯一输入依据，不重新制定时间窗、标签或采样规则。

数据入口直接读取正式Train/Valid清单，继承Clip ID、父视频、Split、原/有效时间、119维target和全部标注来源。不会根据预测重建标签、使用旧缓存、扩展上下文或改动排除清单。Test在本阶段Dataset及诊断运行保护中被明确禁用。

CNN按Proposal初始冻结方案实现：官方ImageNet1K_V1 ResNet-18、移除1000类fc，输出512维。三个独立实例读取同一份项目内官方预训练文件，核对完整SHA256，权重与所有buffer逐元素一致；不下载，不使用旧AnimalAct分类权重。冻结CNN在`model.train()`后仍保持eval，BatchNorm统计不更新。

## 2 统一输入与模型结构

原视频按Manifest真实索引顺序解码；OpenCV BGR明确转换RGB，以PIL bilinear保持宽高比、完整画面短边224，当前`640×360 → 398×224`。使用ImageNet均值/标准差归一化，没有中心裁剪或letterbox。

帧输入`[B,Tmax,3,224,398]`；真实长度`[B]`；有效mask`[B,Tmax]`。Tmax只取当前Batch最大长度；短片段保留全部真实帧，数值padding为0、帧索引为-1，不是新增或重复的视频帧。`padding_mask`与有效mask互补。

```text
release-v1 Clip/真实帧索引/119标签/来源
  → 顺序精确解码 → RGB完整画面等比Resize → Normalize
  → 动态Collate（lengths + valid/padding masks）
  → 只收集sum(lengths)个真实帧
  → 冻结ImageNet ResNet18（每chunk最多8帧）
  → [B,Tmax,512]（无效时间槽为0）
      ├─ Model1: masked mean → [B,512] → Linear → [B,119]
      ├─ Model2: packed BiGRU → [B,Tmax,256]
      │          → masked mean → [B,256] → Linear → [B,119]
      └─ Model3: 相同packed BiGRU → [B,Tmax,256]
                 → 类别对应masked temporal attention
                 → [B,119,256] → 对应分类权重 → [B,119]
```

BiGRU使用hidden128、单层、双向，`pack_padded_sequence(enforce_sorted=False)`按真实长度运行，`pad_packed_sequence`恢复原Batch顺序。没有新增Transformer、Spatial Attention、额外Backbone或Dropout结构。

Model3对每个类别c计算`score[b,c,t] = A[c]·h[b,t]`，无效时间步置负无穷，再沿时间softmax；`z[b,c] = Σt alpha[b,c,t] h[b,t]`；`logit[b,c] = W[c]·z[b,c]+bias[c]`。Attention权重为`[B,119,Tmax]`，不是119类之间的softmax。三个模型输出独立119维logits，支持BCEWithLogitsLoss。

| 模型 | 总参数 | 可训练时间头/分类参数 | 冻结CNN参数 |
| --- | ---: | ---: | ---: |
| Model1 | 11,237,559 | 61,047 | 11,176,512 |
| Model2 | 11,700,151 | 523,639 | 11,176,512 |
| Model3 | 11,730,615 | 554,103 | 11,176,512 |

Model2/3的BiGRU也按相同seed42初始化并核对一致。结构不同的分类部分不声称逐元素一致；三个模型确实使用完全相同的帧、119标签和CNN初始化。

## 3 自动化与真实数据验证

16项自动化契约测试全部通过，详见`unit_test_results.txt`。覆盖：短Batch不会强制扩到16、错误长度/空间形状拒绝、RGB与左右边缘保留、解码非法索引拒绝、Test入口阻断、AP同分组与无正例、31类计数/BCE/Valid阈值、CNN初始化一致、冻结BN、单帧/无序长度、padding不变性、Attention时间归一化/无效权重归零、各头梯度与padding零梯度、零长/非连续mask拒绝。

真实数据固定取每Split三条，均保留完整来源，未用Test：

| Split | 父视频 | 实际输入长度 | 119标签中的正类 |
| --- | --- | ---: | --- |
| Train | TUTKIDMX | 1 | Sensing |
| Train | BIFKWBQT | 10 | Sensing |
| Train | PMQTXPLS | 16 | Eating、Walking |
| Valid | OWJOHCIW | 1 | Jumping |
| Valid | ESQXPYTZ | 10 | Keeping still、Sensing、Shaking Head、Walking |
| Valid | BERDYVEI | 16 | Doing A Face Dip、Eating、Preening、Sensing、Walking |

每个真实Batch为`[3,16,3,224,398]`，lengths均为`[1,10,16]`，只有27个真实帧、21个无效数值槽。另有自动化短Batch`[1,3,10]`实际输出Tmax10，证明没有统一补齐16。

六条Clip分别核验源视频/JSON的SHA，用另一套连续`read()`解码逐帧比对grab/retrieve结果，所选RGB像素完全相同，实际时戳与k/FPS最大误差约`8.88e-16`秒。完整Clip ID、索引、正类和annotation UID/Pointer保存在`validation_results.json`。

三个模型均完成真实Train与Valid前向，输出`[3,119]`且有限；CPU的Train执行backward但不更新参数，CUDA的Train各执行**一次**SGD诊断更新（LR1e-3、BCEWithLogitsLoss）；Valid全部eval/no_grad，不用于反向或更新。所有可训练参数均有有限非零梯度且发生更新；CNN及BN状态完全不变。没有保存更新后的模型或分类头。

额外核验：三个模型CPU帧特征SHA一致；真实RGB padding槽即使替换成NaN也不会进入CNN或改变Model1输出；全部三个时间头在追加NaN padding后输出不变，单独运行真实长度子序列与packed Batch结果一致。Model3无效时间权重严格为0，各类别时间权重和误差最大约`2.38e-7`。CPU与CUDA logits最大绝对差分别约`2.76e-6`、`1.11e-6`、`2.86e-6`。

## 4 RTX5060 8GB资源实测

环境：Python3.13.15、torch2.14.1+cu130、torchvision0.29.1+cu130；NVIDIA GeForce RTX5060 Laptop，PyTorch报告显存约7.96GiB，驱动591.76。本次使用FP32，关闭TF32以检查CPU/CUDA数值一致性，无需混合精度或梯度累积。

| 模型 | CPU诊断秒 | CUDA诊断秒 | PyTorch分配峰值MiB | 保留显存峰值MiB |
| --- | ---: | ---: | ---: | ---: |
| Model1 | 2.149 | 0.721 | 377.28 | 432 |
| Model2 | 1.322 | 0.232 | 382.74 | 500 |
| Model3 | 1.478 | 0.171 | 383.46 | 502 |

CPU包含Train前向/反向及Valid前向，Model1另含真实padding污染检查；CUDA包含模型/数据迁移、Train前向/反向/一次更新及Valid前向。计时受首次CUDA初始化和额外检查影响，**不是模型速度排名或稳定训练吞吐基准**。PyTorch峰值不包括驱动/桌面/其它进程显存；这里只认证B3、Tmax16、chunk8的已测配置，不能据此保证任意更大Batch。

整套小批自动化及真实诊断约9.68秒（含16项契约测试约0.54秒），不含代码编写、报告与历史文件保护核验。最终文件数量、磁盘占用及封存耗时见`FINAL_VERIFICATION.json`。

## 5 评价接口与遗留事项

`evaluation.py`独立提供固定31类mAP、Macro-F1、119类BCE，以及119逐类AP/正Clip数/TP/FP/TN/FN/P/R/F1。沿用Valid31类Macro-F1选阈值的0.05–0.95、0.01网格，并列取低阈值。AP无正例明确为None，完整31类支持不足时不报告完整mAP；本轮没有把6条功能样本分数当性能结果，没有调阈值。

仍需在后续获批阶段落实正式训练协议、训练入口、checkpoint选模及完整Train/Valid评价，本阶段未启动这些工作。SGD/LR1e-3仅用于一次参数更新证明，不是已锁定的正式优化器或学习率。当前不提供已训练的新模型，不声称识别性能改善。

数据限制保持原状：Defensive Rearing仍无可用训练正例；极短Clip无法补出真实运动；既有63个Clip×类别采样覆盖警告不因此消失；同父视频相关Clip、P03等同源风险仍需在后续分析中披露。不得通过改标签、重复帧、重划Split或筛掉主类解决这些问题。

当前release统一空间尺寸可直接Collate；未来如引入不同长宽比，需要独立空间分桶接口，当前会明确报错，不静默裁剪/拉伸。未来如解冻CNN，应另行验证BN政策、显存及训练参数；本轮只验证Proposal的冻结CNN初始路径。

## 6 文件保护与停止

保护核验覆盖36180个旧项目文件大小/修改时间、544份此前Phase0–1.4.3文件内容SHA（含正式release的56个文件），以及官方ResNet18预训练文件SHA。均保持不变。旧大视频/权重不做无关全量重读，六条真实样本的源视频/JSON另外核对内容SHA。详细证据见`protection_baseline.json`及`protection_verification.json`。

所有新增内容仅在`animalact2/phase1_5_1/`。沿用独立.venv，无依赖安装、无全局环境变量修改、无网络下载；未修改AnimalAct1.0、原视频/标注/官方Split或release。运行保护阻断Test媒体/标注/清单和旧任务权重/缓存读取；发布包Test文件仅作历史SHA完整性检查，不进入模型验证。

**完成后停止等待审核。不自动进入Phase1.5.2，不启动正式训练，不提取全量特征、不开发Dashboard。**
