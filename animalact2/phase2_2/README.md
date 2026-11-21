# VideoMAE release-v1 adaptation Pilot

仅100个Train、20个Valid Clip。详细实际结果见 `FINAL_PHASE2_2_REPORT.md`。本目录是技术验证，不是正式训练或识别准确率实验；完成后停止等待审核。

## 输入与适配

复用 `phase1_5_1.ReleaseClipDataset`、严格真实帧解码、RGB完整画面、PIL bilinear短边224等比Resize及ImageNet归一化。保持原始帧索引、119维标签和全部来源。

原始 `[B,L,3,224,398]`（L=1–16）仅在Patch卷积内对齐：右边2列归一化零；奇数L追加1个归一化零时间槽。无中心裁剪、拉伸、窗口扩展、真实帧重复或固定16帧补齐。一个真实帧的末尾tubelet仍有效，真实长度与有效性独立记录。

冻结既有本地Kinetics VideoMAE-Base，空间/时间位置表插值，按相同原始长度分桶提取。输出空间均值 `[ceil(L/2),768]` FP32；它是经过全局时空交互的tubelet特征，不是独立逐帧CNN特征。

## 接口

- `adapter.FrozenVideoMAEAdapter`：冻结eval/inference_mode编码器，返回 `features` 和 `geometry`。
- `data.CachedPilotDataset("train"/"valid")`：读取120条封闭选择的特征及标签来源；禁止Test。
- `data.collate_features`：输出 `[B,Lt_max,768]`、CPU int64 `lengths`、`valid_frame_mask`、`padding_mask`、`[B,119] target`、Clip和父视频ID。
- `heads.VideoMAEProposalHead`：`mean`、`bigru_mean`、`bigru_attention`，输入上述特征、长度及有效Mask。BiGRU每方向hidden=128。

每个Clip缓存目录有 `features.npy`、`receipt.json`，以SHA256核验内容和输入来源；`tubelet_frame_indices` 的null是模型内部数值占位，无原视频帧身份。原子提交，有效缓存拒绝覆盖。receipt保留真实帧数量及标签证据；不存RGB或完整空间Token。

## 已执行入口

从项目根目录运行，沿用现有 `.venv`：

```powershell
# 保护基线：只读元数据/内容校验，写入本阶段目录
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_2.support

# 完成后拒绝重写已通过Pilot；中断时可校验复用已提交的Clip缓存
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_2.run_pilot

# 只读复核历史完整性，任务权重仅按字节哈希，不反序列化/复用
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_2.support verify

# 独立复核已保存缓存、生成报告及交付SHA清单，无模型运行
.\.venv\Scripts\python.exe -u -B -X utf8 -m animalact2.phase2_2.finish
```

运行时保护限制写入本目录，禁用网络、旧任务模型/特征加载和Test内容。预训练权重仅使用 `.cache/videomae_source.json` 指向的既有文件；TEMP、HF运行目录等只在本进程内配置，不改变全局环境。

诊断仅对Train执行三次反向，**零次优化器更新**，没有保存任务头权重。Valid只做前向。禁止由本入口扩大为全量提取、进入正式训练或调用Test。

## 证据与限制

原生尺寸数值等价、136个逐帧/边界扰动、120份实际缓存、三种头部Mask及梯度均有独立记录。主报告列出实际提取时间、逻辑磁盘字节、RAM和PyTorch显存峰值。源视频的边缘截断、低分辨率和动作证据不足不会因技术检查通过而被修复；插值及部分有效tubelet对分类性能的影响尚未验证。
