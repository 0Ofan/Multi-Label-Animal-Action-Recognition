# AnimalAct 2.0 Phase1.5 完整训练前准备报告

**全部训练前准备完成，现停止等待人工审核。未启动正式训练，未读取Test源视频、标注或预测，未提取全量特征。**

## 1 复用与实现边界

以正式`animalact2-2.0.0-release-v1`为数据依据，复用且不修改Phase1.5.1已经通过验证的Dataset、真实帧Decoder、完整画面Transform、动态Collate、冻结ResNet18及三个模型。新代码统一放在`animalact2/phase1_5/`，没有复制重写这些已有核心功能，没有导入AnimalAct1.0任务权重或额外模型组件。

保留Model1 ResNet18+Mean、Model2 ResNet18+BiGRU+Mean、Model3 ResNet18+BiGRU+Class-specific Temporal Attention。CNN初始化、帧表、预处理、119位标签及真实长度/mask完全统一；CNN保持冻结，BN保持eval，BiGRU使用packed sequence，Mean/Attention屏蔽padding。研究结构和参数量与Phase1.5.1一致。

## 2 全量Train/Valid检查

| Split | 正式Clip | 实际涉及父视频 | 失败 |
| --- | ---: | ---: | ---: |
| Train | 4667 | 2063 | 0 |
| Valid | 979 | 442 | 0 |
| 合计 | 5646 | 2505 | 0 |

按父视频顺序完整解码一次，覆盖所有Clip的实际选帧，避免为重叠窗口重复解码同一父视频。共成功解码**326795帧**，对**73884个唯一采样帧**实际执行RGB、PIL bilinear全画面Resize及有限值/形状检查；没有将全部RGB写出。源视频及JSON SHA均与正式元数据一致，路径、帧数、FPS、全部Clip帧索引、有效时间、Mask及标签来源通过核对，时戳最大误差约`1.42e-14`秒。

逐父视频重新读取原Train/Valid JSON，按批准的终点截断、半开区间、正交集并集规则独立核对119位标签、annotation UID/Pointer、密集和抽样支持。不是只验证路径存在或只数Manifest行。保留Train43、Valid9个“有密集支持但均匀采样漏过”的既有标签对，没有改采样或静默过滤。

全量检查约**94.20秒**，4个CPU线程；详细证据见`data_audit/summary.json`、`clip_checks.csv`、`parent_video_checks.csv`、`errors.json`，119类和31主类支持见同目录统计表。只处理正式Manifest涉及的2505个父视频，未将16个没有正式Train/Valid Clip的官方父视频重新纳入。

## 3 共享训练与评价框架

已实现统一训练/Valid循环、AdamW、按实际样本数归一化的梯度累积、梯度裁剪、fixed及warmup+cosine调度、Valid mAP选模、早停、种子与配置管理、历史日志和模型恢复。按seed+epoch确定性shuffle，使相同seed下三模型逐轮输入顺序一致。末尾短microbatch和不足一个完整累积组的情况均正确处理。

评价复用原有固定31类口径：mAP、Macro-F1、119维BCE，逐类AP/正Clip数/正父视频数、TP/FP/TN/FN/Precision/Recall/F1及31个one-vs-rest 2×2混淆矩阵。小批缺主类正例时，不伪造完整31类mAP。正式模型按完整Valid mAP选择；最终阈值只在最佳模型的完整Valid上按0.05–0.95、0.01网格选择，并列取低阈值。没有Test入口。

每个正式实验会保存`last.pt`、`best.pt`、配置/来源指纹、CSV/JSON/JSONL训练记录、逐类及混淆矩阵表、Valid预测和最终结果。当前没有正式训练曲线；已备好曲线所需字段。Train曲线明确是更新前在线累计分数，不能称为同一个checkpoint上的精确Train重评估。

为减少重复磁盘占用，Checkpoint保存全部任务头、AdamW、LR、早停、历史、RNG和类别策略，并引用冻结CNN的官方完整SHA，而不重复嵌入CNN。加载时验证原ImageNet文件、配置、Train/Valid及实现指纹。恢复包括Python/NumPy/Torch/CUDA随机状态；诊断checkpoint有显式标记，正式训练拒绝加载。

## 4 BCE、ASL与Defensive Rearing

两种损失均可运行。BCE为首轮统一基线；ASL参考为gamma_neg4、gamma_pos0、clip0.05，focal权重参与完整梯度。测试验证了ASL极值稳定、简单负样本抑制，以及gamma=0/clip=0时与BCE一致。真实小批ASL结果仅用于功能验证，不作为损失优劣证据。

Train分布：1类零正例，20类只有1个正Clip，53类只有1–5个，63类只有1–10个，中位数7；Sensing1388、Keeping still1247、Walking768。长尾真实存在，但首次三模型比较先保持BCE，以免同时改变结构和损失。ASL后续是否值得用需独立受控实验，当前不执行。

唯一零Train正例是Defensive Rearing（ID23）。保留类别定义、119维标签和logits，但不纳入训练目标；每步恢复其专属分类/Attention参数行，防止AdamW decay造成更新，推理标签函数不将其作为已学会动作输出。119类BCE仍如实另报，同时给出118个Train支持类别BCE；31主类完全不变。所有支持判定只取Train，不使用Valid或Test决定可训练类别。

## 5 自动化与真实接口验证

**共31项测试通过**：复用Phase1.5.1的16项，加13项共享训练框架测试和2项最终方案/执行门控测试。覆盖输入/颜色/Mask/packed GRU、Attention、loss梯度、Train专属支持、AdamW行保护、可恢复shuffle、LR端点、早停、默认入口不训练、非等长microbatch与末组梯度累积、Checkpoint/RNG/优化器精确恢复及拒绝诊断/错配置恢复。

另完成三模型×BCE/ASL共**6组真实功能诊断**。每组8条Train（长度1/2/3/5/10/15/16/16）、4条Valid（1/3/10/16），仅一次Train参数更新，Valid无梯度。六组输入Clip顺序和target SHA一致，全部输出`[4,119]`且有限，所有可训练参数张量发生更新，零样本专属行与冻结CNN/BN保持不变。

六组Checkpoint重建后的Valid logits均**逐元素一致**。小批主类支持不足，记录的完整31类mAP均为None，没有用小批指标选模型或阈值。每组导出逐类/混淆矩阵表证明接口完整，路径明确标记`partial_valid_functional_only`，不属于研究结果。

资源基准另使用32条Train、16条Valid，每模型仅一次累积更新。合计9次真实数据诊断更新，另有单元测试中的合成梯度检查；不存在epoch正式训练循环。所有保存状态在`diagnostics/`，不能作为正式候选或初始化。记录见`preparation_validation.json`和对应诊断目录。

## 6 实际资源与第一轮配置

RTX5060 Laptop、约7.96GiB、现有CUDA环境，FP32可运行最满`[8,16,3,224,398]`。三个模型PyTorch分配峰值分别约**597.12、602.42、602.77 MiB**，最大保留显存**624 MiB**；不含驱动/桌面等显存，不能直接当总显存占用。

两个worker首批启动约8.37秒，后续纯数据吞吐优于零worker；小样本完整流水线外推每轮Train+Valid约6.2–6.3分钟。整体工程诊断约57.24秒，不含94.20秒全量审计、编写和最后保护检查。

首轮正式方案统一为：seed42、BCE、冻结CNN、microbatch8、累积4（有效32）、AdamW峰值LR1e-3/WD1e-4、2轮warmup+cosine至10%、最多50轮、patience10、Valid31类mAP选模。每完整Train epoch584个microbatch、146次更新。三模型共享协议hash一致，没有单独调参。

三模型都跑满50轮的名义外推约**15.6小时**；保守短基准误差范围约10.9–31.1小时，实际早停会减少成本。在线解码和反复运行冻结CNN是当前成本来源；本阶段按约束没有生成全量特征缓存，也没有将不存在的缓存作为可运行前提。完整理由及参数见`FIRST_ROUND_TRAINING_PLAN.md`、`first_round_plan.json`及`configs/`。

## 7 文件保护、遗留与停止

所有新文件仅在本目录。初始保护记录36205个旧文件大小/修改时间，543个非Test历史文件内容SHA，并核对官方ImageNet文件SHA；完整最终状态、文件数及占用见`FINAL_VERIFICATION.json`。Test源文件及分区Manifest不打开；Test文件只核对大小/修改时间，避免以保护检查为由读取Test标签。release-v1、Phase1.5.1、AnimalAct1.0及原始数据均未修改，没有依赖安装、全局环境修改、下载、全量特征提取或Dashboard。

遗留事项是正式训练及完整Train/Valid性能评价本身，本轮依要求不执行。技术验证通过不证明新任务已经获得高mAP；短动作、43/9个采样覆盖警告、极少样本类别和P03等来源风险仍保持原状。后续如要延长预算或比较ASL，应按统一受控协议另行审核，不能按Test表现调整。

**完成并停止，等待人工审核。正式训练入口默认只预检；本轮未使用`--execute-training`，不自动启动后续任务。**
