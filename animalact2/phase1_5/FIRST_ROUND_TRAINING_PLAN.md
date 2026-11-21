# 第一轮正式训练方案（待审核，未执行）

## 研究比较与固定输入

只比较Model1/2/3：外观均值、BiGRU时间建模、类别时间注意力。三个模型直接复用Phase1.5.1独立实现，不加入新结构。所有正式实验使用release-v1的4667个Train、979个Valid Clip，同一RGB完整画面、短边224、PIL bilinear及ImageNet归一化；帧索引/119维target/真实长度/mask均直接继承正式Manifest，没有新采样、重复帧、上下文或增强。

官方ImageNet ResNet18初始化完全一致，CNN全冻结、BN始终eval，帧chunk=8。M2/3仍为hidden128、单层双向GRU。每轮每个Train Clip恰好访问一次，按seed+epoch确定性shuffle；不额外改变父视频或类别采样权重。不同窗口的相关性及P03等既有风险保持披露。

## 统一参数

| 项目 | 第一轮设置 |
| --- | --- |
| 配置 | `configs/model{1,2,3}_seed42_bce.json` |
| 随机种子 | 全部42；123/2026不在本轮自动执行 |
| 输出 | 119个独立logits |
| 目标 | BCE，118个有Train正例类别参与；23号保留但标unsupported |
| microbatch | Train/Valid均8，动态时间padding |
| 梯度累积 | 4；正常有效batch32，末组按实际样本数归一化 |
| DataLoader | 每个loader 2 workers、prefetch1、pin memory、persistent workers |
| 优化器 | AdamW，峰值LR1e-3，weight decay1e-4，梯度范数裁剪5 |
| 调度 | 前2轮warmup：5e-4→1e-3；之后cosine，第50轮降至1e-4 |
| 预算 | 每模型最多50轮，每轮584个Train microbatch、146次参数更新；最多7300次更新 |
| Early Stopping | 连续10轮Valid mAP不刷新，min_delta=0；各模型上限相同，实际早停轮数可不同 |
| 精度 | FP32，TF32关闭；当前显存足够，无需AMP/梯度检查点 |
| 选模 | 完整Valid的固定31类mAP最高；并列保留较早checkpoint |
| 阈值 | 最佳模型的完整Valid上最大化31类Macro-F1，0.05–0.95步长0.01，并列取低阈值 |
| Test | 本轮不读取、不预测、不调参、不选模 |

Epoch日志同时记录训练目标、在线Train mAP/F1/BCE、Valid mAP/F1/BCE、LR、更新数、耗时及早停状态。Train在线分数由该轮更新前的不同参数状态累计，不冒充固定checkpoint在Train上的重新评价；Valid始终是该轮结束的固定模型。最终F1采用Valid选出的阈值，与epoch0.5阈值曲线分开。

## 为什么第一轮选择BCE

Train标签分布明显长尾：119类中1类无正例，20类只有1个正Clip，53类只有1–5个，63类只有1–10个；中位数7，最大Sensing1388。多Clip仍可能来自同一个父视频，正Clip数不能当作独立拍摄来源数。

长尾使ASL值得保留，但不能将AnimalAct1.0不同任务的损失实验收益直接转移到2.0。第一轮保留Proposal的BCE严格架构比较，先观察新任务的学习与过拟合曲线。ASL（gamma_neg4/gamma_pos0/clip0.05）已经实现并通过功能测试，参考配置保持其它设置一致，但本阶段和本轮计划不自动启动额外损失消融。

Defensive Rearing无Train正例，不能被当作正常学会的类别：保留119维标签/输出，不训练其专属行、不让weight decay改变它，部署标签过滤中标为不可输出。训练loss归一化为118类；评价另给原始119类BCE，并保持31主类完全不变。

## 实测资源和预计成本

全量CPU数据检查约94.2秒；这不是每轮训练时间。小样本端到端基准采用32 Train、16 Valid、已启动worker，包含真实解码、Resize、GPU搬运、冻结CNN、任务头前向/反向及一次累积更新，没有特征缓存。

| 模型 | 实测32条Train秒 | 实测16条Valid秒 | 外推完整Train+Valid一轮 | PyTorch分配峰值 |
| --- | ---: | ---: | ---: | ---: |
| Model1 | 2.04 | 1.34 | 约6.33分钟 | 597.12 MiB |
| Model2 | 2.00 | 1.29 | 约6.17分钟 | 602.42 MiB |
| Model3 | 1.98 | 1.34 | 约6.17分钟 | 602.77 MiB |

显存包括额外的最满`[8,16,3,224,398]`batch测试；最大保留显存624MiB。它是PyTorch统计，不含驱动、桌面和其他进程。首批worker启动另约8.4秒；后续纯数据读取约0.034秒/Clip，0 worker约0.062秒/Clip。启动时间和prefetch短样本都会影响外推，不据此声称稳定FPS。

三个模型顺序执行、全部达到50轮的名义估算约**15.6小时**；按短基准误差给出宽范围约**10.9–31.1小时**，早停将减少成本。CPU解码、在线重复运行冻结CNN、供电/温度及系统缓存都会改变实际值。该估计不是已经发生的训练耗时；实际每轮必须记录后再核对预算。

选择统一50轮、patience10是首轮资源控制，不是按任何模型分数调整。若学习曲线表明预算不足，应在审核后对三模型统一延长并发布新的对比协议，不能只给某个模型更多机会。当前不提取全量特征缓存，也不把不存在的缓存视作训练前提。

## 验收与停止

训练准备已经通过全部Train/Valid数据检查、31项自动化测试、三模型BCE/ASL真实小批更新、Checkpoint及RNG/优化器恢复、Mask及8GB资源检查。三个入口默认只预检；必须显式执行训练命令才会训练。所有初始正式模型将重新从官方ImageNet和seed初始化，禁止加载本阶段诊断权重。

**本方案待人工审核；当前没有正式训练，没有Test推理。**
