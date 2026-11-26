# 高精度定位：第一阶段开发记录

目标：至少 80% 的正查询，其最高分区间的起点、终点同时落在同一个真实事件边界的 ±0.5 秒内。此为开发目标，尚未达到或独立验证。

## 已核验的数据

仅检查本地 `../AnimalAct` 的 Train/Valid：2,521 个视频、6,032 个行为区间；未发现负长度或超出视频一帧容差的区间。区间长度的第 10/50/90 百分位为 0.375/1.725/6.333 秒。它们只能证明时间字段合法，不能证明人工标注边界准确。程序校验原始标注 SHA256。

已制作并查看 AMVXZFXO（Tit Bird / Sensing）和 AJDDHWBO（Golden Eagle / Walking）在标注起止附近的六帧联系图，保存在 `.runtime/boundary-audit-*.jpg`。鸟类姿态/位置变化可见，但静态联系图不足以确定 Sensing 的准确起点，也不足以判断动作是否在视频结束时真正结束；这两条标注的终点都贴近视频末尾。不得据此认定标注达到 ±0.5 秒精度。更大样本的连续播放复标及标注者一致性检查仍待完成。

## 最小实现

`boundary_grounding.py` 复用现有官方 ImageNet ResNet-18 编码器，以每秒 4 帧为目标顺序解码并保留解码器实际时间戳，不复制帧。模型使用查询动物/动作嵌入、可训练 BiGRU 和活动/起点/终点三个监督通道，另有查询存在性输出。重叠标注先合并；解码活动片段并用边界分数细化，允许多个区间或空结果。

起止边界使用相邻实际采样时刻的中点构成的时间格。0.25 秒目标间隔不是误差保证，低帧率视频的实际间隔可能更大。当前特征提取分批，但 GRU 处理整段特征；超长视频尚未实现带上下文的分块推理，不应声称任意长度支持。

每个正查询增加一个确定性采样的未标注动物/动作组合作为负查询。未标注不等于确定不存在：这一假设需要抽查。暂仅支持词表内动物及模板查询，未知动物在新边界推理中明确拒绝。

验证统计 mIoU、R@1@0.5/0.7、边界同时命中率、漏检率和负查询误报率，并保留逐查询起止误差及整视频基线。漏检计入边界命中率分母。按边界命中率优先、mIoU 次优、负查询误报率再次排序选模型，阈值暂固定 0.5，不使用 Test。

## 可运行命令

在 Project_Test 目录运行；全量命令须另行确认。缓存与权重仅写入新目录，已有分类与定位结果不覆盖。

```powershell
# 小样本流程检查：各取一个 Train/Valid 视频，只更新一轮，不作精度结论
.\.venv\Scripts\python.exe -B -m proposal.boundary_grounding ../AnimalAct --mode smoke --output .runtime/boundary-smoke-review --rate 4 --epochs 1

# 全量特征提取：只读 Train/Valid
.\.venv\Scripts\python.exe -B -m proposal.boundary_grounding ../AnimalAct --mode extract --output proposal/boundary_results --rate 4

# 使用相同缓存训练定位头：只用 Valid 选择模型
.\.venv\Scripts\python.exe -B -m proposal.boundary_grounding ../AnimalAct --mode train --output proposal/boundary_results --rate 4 --epochs 40

.\.venv\Scripts\python.exe -B -m unittest tests.test_boundary_grounding tests.test_proposal tests.test_i18n tests.test_portable
```

本地 4fps 采样上限估算约 55,565 帧，512 维 float32 特征约 114 MB（十进制），另有缓存元数据、模型和运行内存。当前 `.venv` 为 PyTorch 2.14.1 CPU，CUDA 不可用。小样本端到端约 9 秒，含导入与数据读取；不可线性当作全量耗时保证。全量提取和 40 轮训练应按可能数小时安排，具体取决于 CPU 和解码速度。

第一阶段验证：28 项单元/回归检查，27 项通过、1 项可选 VideoMAE 实模型测试跳过；两视频真实 ResNet 小样本训练/验证流程通过。小样本验证正查询只有一条且未命中，仅证明流程运行，不能解释为性能结果。未全量提取或训练。

## 推理与上线边界

`ProposalPredictor.predict_boundaries(video, query, checkpoint_path)` 提供显式实验性推理入口，复用同一个 ResNet 编码器；返回实际采样时间、时间格及候选区间。拒绝 smoke 权重。正式模型训练、验证与人工验收完成之前，不替换现有网页中的定位模型，也不宣称已达到 ±0.5 秒。

独立验收视频尚未建立。历史 Test 已用于探索，不应重新包装为从未查看的盲测。网页接入及验收在训练结果可审核后进行。

## 第二阶段：全量提取与 40 轮训练结果

已按确认的两项命令完成 Train/Valid 的全量提取和 40 轮 CPU 训练，进程正常退出。共 2,521 个视频、55,565 个采样帧，缓存实际为 119,319,024 字节。仅写入 `proposal/boundary_results`，未读取 Test 视频/标注，未覆盖历史结果或替换网页模型。

最优检查点为第 3 轮。原始 Valid 含 809 条正查询和 809 条按未标注组合构造的负查询：

- 双边界同时在 ±0.5 秒内：37.70%（305/809），未达到 80% 目标；整视频基线为 44.13%。
- mIoU：51.74%，整视频基线为 61.20%。R@1@0.5 为 49.20%，R@1@0.7 为 41.04%。
- 正查询漏检 137/809（16.93%），负查询误报 117/809（14.46%）。负查询依赖标注完整性假设，不能视作独立人工核实的无目标测试。
- 672 条有返回结果的正查询中，648 条最高分区间仍是整视频（96.43%）。

验证过程允许未知动物使用零嵌入，而显式推理接口拒绝训练词表外动物。若按实际接口策略，将 58 条未知动物正查询作为拒识/漏检，并拒绝对应 58 条负查询，全部正查询的边界命中率为 33.62%，mIoU 为 46.60%，漏检率为 23.73%，负查询误报率为 13.72%。两种口径分别保存，不能把原始验证数字直接描述为实际接口精度。

`sampling_audit.json` 核验时间格的第 50/90/99 百分位及最大宽度为 0.25/0.25/0.375/0.375 秒。若已知正确标注并选理想时间格，99.13% 的正查询存在满足 ±0.5 秒目标的可用区间；这只是几何上限，不是模型精度。

复核：重新加载保存的权重，对完整 Valid 复算，指标与检查点一致、所有预测区间与保存记录完全一致。对真实视频 AAPREPTK 的 Chirping、Keeping Still 两条查询，在线重新解码/提取的推理结果与缓存评估一致。14 项相关测试通过。详细证据见 `boundary_results/verification.json`，逐轮指标见 `boundary_results/results.json`。

当前确定的解码限制：先将活动概率按 0.5 阈值切为连续段，然后只在每段两端附近各一格调整边界；当活动段塌缩为整视频时，起止分数无法直接生成位于视频中部的短区间。它是否为退化的主要原因，仍需对照实验验证。下一步最小路径是先使用现有权重，在 Valid 上检查独立起止候选生成与排序，比较同一评估口径；再决定是否需要修改损失或重新训练。本轮不追加实验。

结论：全量实验和复现验证已完成，精确定位目标尚未实现。本检查点保持实验用途，未提升为网页默认模型。

## 第三阶段：自动动作与时间区间，实施完成、待全量训练

根据确认的新需求，网页不再要求选择动作或填写动物/动作查询。新检测头沿用 ResNet-18 → BiGRU 方法，对每个时间单元同时预测 119 类的活动、起点和终点，以及视频级存在性。重叠的同类标注跨动物合并；同一动作可多次发生，不同动作可并发。不自动推断动物种类或个体身份。

起止峰值在整个序列中生成候选，不再局限在活动段的两端；按动作分别抑制重复/包含区间。分类与检测独立，检测不会被整视频分类的阈值结果筛掉。输出是动作、起点、终点和未校准分数。

逐文件改动及用途：

- `proposal/action_detection.py`：自动检测模型、复用 4fps 缓存的分批训练、独立边界解码和事件/定位指标；最多 40 轮，Patience 8，默认单 CPU 线程。小样本和历史权重不能代替正式自动检测模型。
- `animalact2/dashboard/ui/services/resources.py`：对现有模型推理结果自动附加时间检测结果；只加载完成训练、非 smoke 且复核启用的检测器。
- `animalact2/dashboard/ui/services/state.py`：上传识别不传手动查询，更新结果身份以使旧查询结果失效。
- `animalact2/dashboard/schema.py`：校验动作类别、分数和时间区间边界。
- `animalact2/dashboard/ui/services/presentation.py`：增加时间区间 CSV，完整 JSON 保留全部检测结果。
- `animalact2/dashboard/ui/pages/video.py`：移除查询与手动输入时间窗口，使用完整上传视频；未就绪时明确提示。
- `animalact2/dashboard/ui/pages/results.py`：接入自动动作区间显示。
- `animalact2/dashboard/ui/components/ui.py`：动作区间列表、时间轴及区间 CSV 下载；空结果和未就绪状态分别处理。
- `animalact2/dashboard/ui/i18n.py`：同步上述中英文文案。
- `tests/test_action_detection.py`：覆盖并发/重复动作、视频中部边界、短动作存在性、掩码、事件匹配、CSV/JSON、推理路由和双语结果展示。
- `tests/test_i18n.py`、`tests/test_proposal.py`：调整旧查询输入/会话状态测试，适配自动识别流程。
- `README.md`：更新使用方式和正式训练命令。

小样本流程只用 2 个训练视频和 2 个验证视频，一次优化；3 次检查均跑通，未作精度结论，且 smoke 权重未启用。最终相关回归检查为 36 项，35 项通过、1 项可选 VideoMAE 实模型测试跳过。浏览器已确认查询框移除、未就绪提示正确，8502 服务已重启加载新版。时间列表和时间轴使用受控测试数据通过双语 AppTest 检查；当前没有正式自动检测权重，不能声称网页已经能可靠输出真实区间。

下一步待确认的全量命令（复用现有只读缓存，无需重新提取特征）：

```powershell
.\.venv\Scripts\python.exe -B -u -m proposal.action_detection ../AnimalAct --mode train --cache proposal/boundary_results/features --output proposal/action_detection_results --epochs 40 --patience 8 --batch-size 8 --threads 1
```

结果写入新目录。只用 Train/Valid；记录逐轮指标，按边界事件 F1 选择模型，连续 8 轮未改善早停。事件指标采用同类一对一匹配，重复误报和漏检均计入；另外报告按视频/动作归组的 mIoU、R@1@0.5/0.7，不能与旧动物/动作查询口径直接混用。训练完成也不自动启用网页权重，先复核结果再决定是否切换。±0.5 秒精度仍需独立验证，长视频资源开销及标注完整性仍是风险。

## 第四阶段：自动检测全量训练及启用决定

确认命令已完成。使用原有 Train/Valid 密集缓存、批量 8、单 CPU 线程；最多 40 轮，在第 37 轮因连续 8 轮无提升而早停。最优检查点为第 29 轮。未读取 Test 视频或标注，未改动历史分类及查询定位权重。

默认阈值 0.5 的 Valid 有 1,040 个合并事件、793 个视频/动作正查询，输出 1,454 个候选区间：

- 起点、终点均在同一事件的 ±0.5 秒内，且动作类别正确：220 个。严格边界精确率 15.13%，召回率 21.15%，F1 17.64%。
- 动作类别正确且 temporal IoU ≥0.5：261 个。精确率 17.95%，召回率 25.10%，F1 20.93%。
- 按视频/动作正查询的最高分候选计算：mIoU 32.16%，R@1@0.5 30.77%，R@1@0.7 24.72%。这些单位与早期动物/动作查询不同。
- 710 个候选动作类别不在对应视频的标注中，822 个候选仍覆盖整视频。前者依赖标注完整性，不能全部断言为肉眼确认的误报；但足以表明当前结果不能按可靠精确定位使用。

固定最佳权重，只在 Valid 上诊断 0.5/0.6/0.7/0.8/0.9/0.95 六档阈值。阈值 0.6 的边界 F1 为 17.94%，只小幅改善。阈值 0.95 的严格边界精确率最高，为 25.42%，召回率仅 8.75%。阈值诊断未改动检查点或默认阈值，也不属于独立测试。

完整 Valid 重新推理后，指标与检查点一致，逐视频区间与保存记录完全一致。真实视频 AJDDHWBO、AKFJYBGG 重新解码和提特征的推理结果与缓存一致；分数数值误差在 1e-6 内。共享输出 schema、CSV/JSON 导出通过检查。

文件：`action_detection_results/detector.pt` 为最优权重；`results.json` 保留 37 轮历史和未启用决定；`validation_records.json` 为逐视频预测；`verification.json` 为复核；`threshold_diagnostics.json` 为阈值诊断；`inference_examples.json/csv` 是两个真实验证视频的实验性候选，供审核，不是已确认的正确动作区间。

启用决定：精度未达标，`enabled_for_inference` 保持 false，网页仍显示自动定位未就绪，分类功能继续可用。代码路径能自动产生不同动作及候选时间，但这不等于已实现可靠定位。本轮只增加训练、诊断、示例及文档记录，没有追加架构改动或新训练。

后续建议先在少量连续视频中复核动作标注与边界，将类别误报和边界误差分开诊断，并验证前景/背景监督及图像特征对动作运动信息的区分能力；再决定是否需要调整损失或使用局部视频特征。增加训练轮数或提高阈值尚不足以解决已观察到的问题。后续实验需另行确认。

## 第五阶段：损失诊断与 16 个 Train 视频对照

只读诊断发现，旧损失把活动、起点和终点混合归一化。Train 正目标质量中活动约占 85%，起点和终点合计约 15%；负目标质量约 98.53% 来自整段视频不存在的动作类别。训练集边界 F1 也只有 34.56%，不是仅在 Valid 上失效。Valid 中，存在动作类别的区间内平均活动分数为 0.668，区间外仍为 0.631，时间区分很弱。

使用真实标签替换单个输出的诊断显示：只替换动作存在性时，Valid 边界 F1 为 25.99%；只替换起止边界时为 70.16%；全部使用理想输出时为 98.78%。这些是使用真实标签的误差定位实验，不是模型成绩，不能部署或用于宣传泛化能力。

根据确认方案，仅调整训练目标，模型结构、解码、标签和历史权重均保留：

- `proposal/action_detection.py`：三条通道分别归一化；将存在动作类别中的时间背景与不存在类别的背景分组，各自归一化；存在性目标忽略 padding。`load_records` 新增可选 splits 参数，默认仍读 Train/Valid，小集检查明确只读 Train。
- `tests/test_action_detection.py`：新增通道/背景梯度权重、padding/空分组及 Train-only 不读取 Valid/Test 标注的回归检查。
- `proposal/overfit_check.py`：新增可复现的 16 视频新旧损失对照命令，保留冻结的旧损失公式。所有小集检查点标记 smoke/diagnostic，推理接口拒绝使用。

运行命令：

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_action_detection tests.test_boundary_grounding
.\.venv\Scripts\python.exe -B -u -m proposal.overfit_check ../AnimalAct --output proposal/action_detection_overfit16 --epochs 200
```

17 项相关测试通过。固定 Seed6116，从不同 Train split_group 中选择 16 个视频，时长 1.5–20 秒，事件均至少 0.5 秒，且至少包含一个距视频两端均有背景的事件。共有 73 个合并事件、46 个视频/动作组合。此选择用于排除明显的采样分辨率限制，不代表完整数据分布。

两组使用相同初始化、相同 batch8 顺序、AdamW LR1e-3/WD0.01、单 CPU 线程、相同解码和阈值 0.5，各最多 200 轮，只在同一批 16 个 Train 视频上检查。预设拟合门槛为严格边界精确率与召回率均达到 90%。

- 相同第 60 轮：旧损失边界 F1 为 21.98%，新损失为 95.24%。
- 新损失第 60 轮达到门槛：74 个预测、70 个严格边界命中；精确率 94.59%，召回率 95.89%，F1 95.24%，训练子集 mIoU 90.15%。
- 旧损失运行满 200 轮仍未达到门槛；最佳第 190 轮精确率 47.50%、召回率 78.08%、F1 59.07%。
- 各自最优检查点上，存在动作类别的区间外平均活动分数从旧损失 0.747 降至新损失 0.136，符合增强时间背景监督的目的。

保存权重重新加载后，所有指标和区间均与记录一致；两组初始预测完全一致；诊断权重的生产推理拒绝检查通过。历史完整模型 SHA256 保持不变，网页启用状态仍为 false。详细结果和校验位于 `action_detection_overfit16/results.json`、`verification.json`，包含固定的视频 ID、标注校验值、逐轮指标及小集预测。

结论：在这批可表达的训练样本上，新损失明显改善了边界拟合，支持继续做全量 Train/Valid 对照；不能据此声称未见视频达到 95% 精度。本轮未全量重训、未读取 Valid/Test 视频或标注，也未更换网页模型。

下一步待确认命令（保留旧全量结果，另写新目录）：

```powershell
.\.venv\Scripts\python.exe -B -u -m proposal.action_detection ../AnimalAct --mode train --cache proposal/boundary_results/features --output proposal/action_detection_balanced_results --epochs 40 --patience 8 --batch-size 8 --threads 1
```

保持原有全量训练设置，仅更换损失，以 Valid 判断改善是否能泛化。未标注行为可能存在、稀有类别样本不足、冻结图像特征的运动辨别能力等风险仍未消除。

## 第六阶段：新损失全量 Train/Valid 对照完成

已执行确认的命令，在 `proposal/action_detection_balanced_results` 写入新结果。训练完成 40 轮，最佳检查点为第 34 轮；仅使用 Train/Valid，未读取 Test 视频/标注，旧完整模型 SHA256 保持不变。模型、采样缓存、初始化种子、批量、优化器及固定阈值与旧完整实验保持一致，主要变量为损失。

固定阈值 0.5 的 Valid：1,010 个候选、1,040 个合并标注事件。187 个预测的动作类别正确且双边界同时在 ±0.5 秒内；严格边界精确率 18.51%、召回率 17.98%、F1 18.24%。旧损失对应值为 15.13% / 21.15% / 17.64%。新损失 F1 增加约 0.60 个百分点，精确率上升但召回下降，单种子结果不足以说明统计显著性。

按视频/动作正查询计算，新损失 Valid mIoU 为 29.31%，低于旧损失 32.16%；R@1@0.5 为 28.75%，R@1@0.7 为 24.34%。事件 temporal IoU≥0.5 的精确率/召回率/F1 为 23.17% / 22.50% / 22.83%。不同指标表现有取舍，不能只挑 F1 的微小提升描述为全面改善。

固定新权重在完整 Train 上复算，边界 F1 为 46.75%，高于旧完整模型的 34.56%；但 Valid 仅 18.24%。存在动作类别的活动分数：Train 区间内/外均值为 0.812/0.206，Valid 为 0.538/0.316。该结果支持监督修正改善了拟合及背景区分，但完整数据拟合和泛化均仍不足；16 视频小集的 95.24% 不能描述为新视频精度。

新 Valid 仍有 608 个候选类别不在对应视频标注内，832 个候选覆盖整视频（82.38%）。未标注不等于人工确认不存在，但误报与长区间偏置仍明显。对固定新检查点作 0.3/0.4/0.5/0.6/0.7/0.8/0.9/0.95 阈值诊断，最佳边界 F1 仅在阈值 0.4 达到 18.61%；阈值 0.95 的最高边界精确率为 43.59%，召回率只有 1.63%。该诊断只用 Valid、未改动原权重及默认阈值，不构成独立验收。

保存权重重新加载后，完整 Valid 指标和全部区间与原记录一致。AJDDHWBO、AKFJYBGG 两个真实视频重新解码/提特征后，候选类别及时间区间与缓存推理一致，分数误差在 1e-6 内；共享 schema、CSV 和 JSON 导出均通过检查。证据见新目录的 `verification.json`、`threshold_diagnostics.json`、`inference_examples.json/csv`。`results.json` 保留逐轮训练与未启用决定。

启用决定：新旧自动检测模型均保持禁用，网页分类继续可用。本轮完成了已确认的全量训练、对照和复核；未更换网页模型、未追加架构改动或其他训练。精确自动定位目标尚未实现。下一步应先验证现有特征对动作变化和边界的可分性，并检查标注完整性及一致性，再决定是否使用预训练视频特征；不建议继续仅凭加轮数或单一阈值调整期待达到目标。

## 第七阶段：按用户确认启用实验性网页预览

用户明确选择先查看实际自动候选结果。该决定改变展示开关，不代表精度已经达标。未重新训练，仍使用新损失第 34 轮权重，文件 SHA256 与既有 verification.json 一致。

本轮修改及原因：

- `animalact2/dashboard/ui/services/resources.py`：网页接入路径切换为 `action_detection_balanced_results/detector.pt`。
- `proposal/action_detection_balanced_results/results.json`：将 enabled_for_inference 设为 true，review_status 记为 enabled_experimental_preview_by_user；保留原始训练指标及未达标事实。
- `animalact2/dashboard/ui/pages/video.py`：模型可用时显示实验性启用提示。
- `animalact2/dashboard/ui/components/ui.py`：结果区增加醒目的“类别和区间可能有误、不是核实标注”提示。
- `animalact2/dashboard/ui/i18n.py`：同步上述中英文文案。
- `tests/test_proposal.py`：分类路由单元测试隔离自动检测依赖，避免启用模型后让 mock 视频触发真实定位；自动检测路由另有测试。
- `README.md`：说明当前实验预览模型路径和状态。

22 项相关测试通过。共享上传/推理/导出流程用真实 Valid 视频 AKFJYBGG.mp4 检查，自动返回 Sensing、Flapping、Shaking、Preening 及多次 Shaking 的 5 个时间区间；未输入动物或动作查询。浏览器中实际上传、点击识别并进入结果页，动作区间表、时间轴和 CSV 下载均可见，实验提示可见。示例仅证明真实预测已呈现，不证明候选正确。

8502 服务已重启加载修改，旧页面需刷新后重新识别。当前分类头和时序检测头独立，候选动作可能与整视频分类列表不同。候选分数未经校准，默认阈值仍保持 0.5，未通过提高阈值隐藏误报。

端到端校验位于 `.runtime/experimental-preview-verification.json`；实际网页截图为 `.runtime/experimental-time-preview-20261123.jpg`。早期 verification.json 中 enabled_for_inference=false 是当时的校验快照；当前启用状态以本阶段说明及 results.json 为准。
