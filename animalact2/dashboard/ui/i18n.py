"""Shared display translations. Widget values and exported model data retain their IDs."""
import streamlit as st

ZH = {
"Video":"视频识别", "Results":"识别结果", "Evaluation":"切片实验评估", "Proposal":"提案实验",
"DASHBOARD":"功能导航", "Dashboard":"功能导航", "Model Information":"模型信息",
"AnimalAct 2.0 · Research Dashboard":"AnimalAct 2.0 动物动作识别",
"Proposal experiments":"提案实验", "Proposal classification model":"分类模型",
"ResNet-18 · whole-video multi-label classification · fixed official splits":"ResNet-18 整视频多标签分类 · 官方固定划分",
"Whole-video classification · 119 trainable classes":"整视频分类 · 119 个可训练类别",
"Template-query temporal-window scoring":"模板查询与时间窗口匹配",
"Test per-class AP and confusion counts":"测试集逐类 AP 和混淆计数",
"Download proposal evaluation":"下载分类评估结果", "Template-query temporal grounding":"模板自然语言时序定位",
"Experimental localization. The learned scorer has not improved over returning the entire video.":"实验性定位：当前模型尚未超过整视频基线。",
"Queries whose best whole-video baseline IoU is below 0.5":"整视频基线 IoU 小于 0.5 的查询子集",
"Download grounding evaluation":"下载定位评估结果", "Run the proposal model":"运行提案模型",
"Unseen Test example":"未见过的测试视频示例", "Upload a video":"上传视频",
"Video for whole-video classification":"选择整视频分类文件", "Optional template query":"模板查询（可选，使用英文）",
"Run proposal inference":"运行提案推理", "Download proposal prediction":"下载预测结果",
"Example query: ":"查询示例：", "Running ResNet-18 on CPU":"正在使用 CPU 运行 ResNet-18",
"The grounding checkpoint is still being prepared.":"定位模型尚未准备就绪。",
"The first interval is the top-ranked prediction. The remaining intervals are alternatives.":"第一个区间为最高分预测，其余为备选区间。",
"The complete whole-video experiment is running. Results appear after evaluation completes.":"整视频实验正在运行，评估完成后将显示结果。",
"Query format: A {animal} is {action}. The highest-ranked interval is evaluated against all reference intervals using maximum temporal IoU.":"查询格式为 A {animal} is {action}.，使用最高分预测与所有真实区间的最大时间 IoU 评估。",
"Model":"模型", "Valid mAP":"验证集 mAP", "Test mAP":"测试集 mAP", "Test Macro-F1":"测试集宏平均 F1",
"Concurrent recall":"并发动作召回率", "Concurrent precision":"并发动作精确率",
"Threshold":"阈值", "VIDEO INTELLIGENCE":"动物视频分析", "AnimalAct 2.0 Recognition":"AnimalAct 2.0 动作识别",
"Upload an animal video and let the AI model recognize multiple actions in the video.":"上传动物视频，识别其中出现的多个动作。",
"Classification head":"分类头", "BiGRU + Temporal Attention (primary)":"BiGRU 与时间注意力（主模型）",
"BiGRU Mean":"BiGRU 平均池化", "Current model":"当前模型", "Backbone":"骨干网络",
"Action classes":"动作类别数", "Real input frames":"实际输入帧数", "Upload your video":"上传视频",
"Drop a file to begin · Up to 200 MiB · Your video stays local":"选择视频开始 · 最大 200 MiB · 在当前服务处理",
"Drop a file to begin · Up to 200 MiB · Processed on this app server":"选择视频开始 · 最大 200 MiB · 在当前应用服务器处理",
"App server processing · Fixed model":"应用服务器处理 · 固定模型",
"Time detection failed; classification results are retained.":"时间定位失败，已保留行为分类结果。",
"Choose an animal video":"选择动物视频", "Clear video":"清除视频",
"Your video preview will appear here":"视频预览将显示在这里", "Animals in motion. More than one action.":"展示动物运动与多种行为。",
"Recognition settings":"识别设置", "Processing device":"计算设备", "Select a clip window":"选择输入时间窗口",
"CUDA is available.":"CUDA 可用。", "CUDA is unavailable. Recognition will run on the CPU.":"CUDA 不可用，将使用 CPU 识别。",
"Window start (seconds)":"窗口起点（秒）", "Window end (seconds)":"窗口终点（秒）",
"Decision threshold":"判定阈值",
"Selected on validation data and kept fixed. Top-ranked actions below this value are not selected predictions.":"阈值由验证集选定，低于阈值的动作不计入识别结果。",
"Recognition complete":"识别完成", "Run Recognition":"开始识别", "Recognizing actions…":"正在识别动作……",
"Preparing the video and loading the frozen model if needed.":"正在准备视频并加载模型。",
"Sequential real-frame decode → full-frame resize → aligned variable-length VideoMAE → selected BiGRU head.":"依次完成帧解码、整帧缩放、VideoMAE 特征提取和 BiGRU 分类。",
"Recognition failed: ":"识别失败：",
"Prediction saved for this video. Open Results for all 119 classes.":"已保存本视频预测，可在识别结果页查看 119 类分数。",
"One clip. Multiple actions.":"一个片段，多种动作",
"A frozen VideoMAE encoder feeds the selected BiGRU head. Results describe the supplied clip, not automatically detected action intervals.":"冻结的 VideoMAE 提取特征后输入 BiGRU，输出所选片段的动作类别。",
"No action onset/offset is inferred. The window bounds are input bounds only. Predictions can be empty or multi-label.":"输入窗口由用户指定。分类输出可以为空或包含多个动作。",
"119 raw class scores are retained. Defensive Rearing has no Train positive and cannot be emitted as a learned action. Scores are uncalibrated; errors remain possible.":"保留 119 类原始分数。Defensive Rearing 缺少训练正例，暂停输出该类。分数未经校准。",
"Recognition preview":"识别预览", "Predicted actions":"预测动作", "Inference time":"推理耗时",
"No action score reached the fixed threshold. Top-K remains available for inspection.":"没有动作达到阈值，仍可查看分数排名。",
"View detailed results →":"查看详细结果 →",
"The whole video is treated as ONE supplied clip. You may select a time window; it is not an automatically localized action interval.":"整段视频作为一个输入片段，也可以手动选择时间窗口。",
"Short clip: all real frames are kept. No repeated frames and no padding to 16; an odd last frame gets one internal zero slot for tubelet alignment.":"短视频保留全部真实帧。奇数帧会增加一个内部零槽以对齐模型输入。",
"MP4/H.264 is recommended for browser playback. Other supported formats may still be recognized if their codec is available.":"建议使用 MP4/H.264 以便浏览器播放。其他格式需系统支持相应解码器。",
"PREDICTION EXPLORER":"预测结果查看", "Recognition Results":"识别结果",
"Explore every action score from your latest video. No additional inference is run on this page.":"查看最近一次视频预测的全部动作分数。",
"Highest score":"最高分数", "Fixed threshold":"固定阈值", "Probability distribution":"分数分布",
"Search actions":"搜索动作", "Search mode":"搜索方式", "Categories":"类别范围", "Sort":"排序",
"Display":"显示数量", "Semantic (LLM)":"语义搜索（LLM）", "Keyword":"关键词",
"All 119 classes":"全部 119 类", "Predicted only":"仅已识别动作", "Main 31 classes":"31 个主评估类",
"Extension classes":"扩展类别", "Search relevance":"搜索相关度", "Highest probability":"分数从高到低",
"Lowest probability":"分数从低到高", "Action name":"动作名称", "Class ID":"类别 ID",
"Top 10":"前 10 项", "Top 20":"前 20 项", "All 119":"全部 119 项",
"Category details":"类别详情", "Select an action":"选择动作", "Export prediction":"导出预测",
"Download the complete prediction with class IDs, scores, threshold and inference time.":"下载包含类别 ID、分数、阈值和耗时的完整预测。",
"Download CSV":"下载 CSV", "Download JSON":"下载 JSON", "Action":"动作", "Probability":"分数",
"Prediction":"预测", "Selected":"已选中", "Not selected":"未选中", "Decision":"判定",
"Why matched":"匹配理由", "Above threshold":"达到阈值", "Eligible":"可输出", "Predicted":"已预测",
"Main class":"主评估类", "Main evaluation class":"主评估类", "Extension class":"扩展类",
"Matched classes and reasons":"匹配类别与理由",
"No class reached the validation-selected threshold. A low score does not establish that an action is absent.":"没有类别达到验证集选定的阈值，低分不能确认动作不存在。",
"Scores are independent sigmoid outputs, not calibrated certainty. No ground truth is provided for this upload, so accuracy, precision and recall are not shown.":"分数为独立 sigmoid 输出，尚未校准。上传视频没有真实标签，因此不显示准确率、精确率和召回率。",
"Semantic search re-ranks the view only; prediction scores, threshold and emitted labels are unchanged.":"语义搜索用于筛选与排序显示结果。",
"No action class matched this query. Try another description or switch to Keyword mode.":"没有匹配类别，请调整描述或使用关键词搜索。",
"LLM not configured · offline alias matching is used. Add [llm] api_key under Streamlit Secrets to enable semantic search.":"尚未配置 LLM，使用离线双语匹配。可在 Streamlit Secrets 中配置 [llm] api_key。",
"No categories match the current filters.":"当前筛选条件下没有类别。",
"Your next discovery starts with a video":"请先识别一个视频",
"Upload a clip on the Video page and run recognition. Your latest prediction will appear here, with all 119 action scores.":"在视频识别页上传文件并运行识别，即可查看全部 119 类分数。",
"Changing pages never starts recognition.":"切换页面不会自动运行识别。",
"Independent sigmoid score":"独立 sigmoid 分数",
"REPRODUCIBLE EXPERIMENTS":"可复现实验", "AnimalAct 2.0 Evaluation":"AnimalAct 2.0 切片实验评估",
"Read-only real Train/Valid results. Test has not been evaluated for this release.":"展示切片版训练与验证结果，切片版尚未进行测试评估。",
"Overview":"概览", "Per-class AP":"逐类 AP", "Confusion":"混淆矩阵", "Training":"训练过程",
"Generalization":"泛化分析", "Three heads · three seeds · Valid mean ± sample std":"三种分类头与三个随机种子的验证均值和样本标准差",
"ResNet18 vs VideoMAE":"ResNet18 与 VideoMAE 对照", "All nine runs · fixed best checkpoints":"九组实验的最佳检查点",
"AP split":"AP 数据划分", "Cross-seed per-class stability":"各类别跨种子稳定性",
"Confusion model":"混淆矩阵模型", "Confusion seed":"混淆矩阵随机种子", "Confusion class":"混淆矩阵类别",
"Training model":"训练模型", "Training seed":"训练随机种子",
"Fixed-checkpoint Train/Valid gap":"固定检查点的训练与验证差距",
"Pred negative":"预测负类", "Pred positive":"预测正类", "True negative":"真实负类", "True positive":"真实正类",
"Precision":"精确率", "Recall":"召回率", "Source: ":"来源：",
"Ready · loads on demand":"就绪 · 按需加载", "Loaded · ":"已加载 · ",
"Frozen encoder · BiGRU · selectable head":"冻结编码器 · BiGRU · 可切换分类头",
"Proposal · whole-video ResNet-18 · 119 outputs":"提案实验 · 整视频 ResNet-18 · 119 类输出",
"COMP6116 · Stage 2\n\nAnimal action recognition":"COMP6116 第二阶段\n\n动物动作识别",
"No placeholder predictions or evaluation values have been substituted.":"请检查服务和模型文件。",
"AnimalAct 2.0 backend unavailable: ":"AnimalAct 2.0 后端不可用：", "This view could not be loaded: ":"页面加载失败：",
"Local processing · Fixed model":"本机处理 · 固定模型", "AI RESEARCH WORKSPACE":"动物行为研究工作台",
"Validation":"验证集", "Test":"测试集", "Train":"训练集", "valid":"验证集", "train":"训练集",
"mean":"平均池化", "bigru_mean":"BiGRU 平均池化", "bigru_attention":"BiGRU 时间注意力",
}
# Longer passages and numeric sentences are translated using their unchanged text fragments.
ZH.update({
"Evaluation unit: action-interval clip. 31 fixed main classes; BCE118 excludes unsupported Defensive Rearing. BCE119 remains a diagnostic. Multi-seed means are not ensembles.":"评估单位为动作区间片段，采用固定 31 个主类。训练损失覆盖 118 个有训练支持的类别，另记录 119 类诊断损失。多种子均值用于统计。",
"ResNet18 has one Seed42 run; VideoMAE has N=3. Different pretraining sources and feature dimensions; this is not an isolated architecture effect.":"ResNet18 使用单个种子，VideoMAE 使用三个种子。预训练来源与特征维度均不同，请按完整方案对照解释。",
"Train curves are online pre-update metrics; Valid curves use the fixed model at epoch end. Curve F1 uses 0.5, unlike final Valid-selected thresholds.":"训练曲线统计训练过程中模型的表现，验证曲线使用每轮结束时的模型。曲线 F1 阈值为 0.5。",
"Each checkpoint is evaluated on both splits using its unchanged Valid-selected threshold. These are distinct from online Train curves.":"各检查点采用验证集选定阈值，同时评估训练集和验证集。",
"Train mAP is approximately 97–100% while Valid mAP is 43–47%. Strong overfitting is reproducible across seeds; this dashboard does not resolve model accuracy limits.":"训练 mAP 约为 97–100%，验证 mAP 约为 43–47%，多个种子均显示明显过拟合。",
"Defensive Rearing (ID23): raw score is shown for traceability, but it has no Train positives and is excluded from emitted labels.":"Defensive Rearing（ID23）在切片训练中没有正例，仅保留原始分数。",
"2,075 Train / 446 Valid / 447 Test videos. All 119 classes trained. Primary metrics use the fixed 31 classes. Concurrent Test subset contains 108 videos. Checkpoints and thresholds were selected on Valid.":"训练 2,075 个、验证 446 个、测试 447 个视频，全部 119 类参与训练。主要指标采用固定 31 类，并发测试子集为 108 个视频。检查点与阈值由验证集选定。",
"Input window [start,end): ":"输入窗口 [起点,终点)：", " seconds · ":" 秒 · ",
" unique real frames · feature shape ":" 个真实帧 · 特征形状 ",
". These are input bounds, not predicted action times.":"。此处显示输入窗口。",
"Showing ":"显示 ", " of 119 class scores. Filters affect the view only; exports always contain all 119 classes.":" / 119 类分数，导出文件包含全部类别。",
"Semantic search · ":"语义搜索 · ", " matched classes. Only the query and action names are sent to the configured API; the uploaded video is never sent.":" 个匹配类别，仅查询和动作名称发送到 API。",
"Semantic search is unavailable (":"语义搜索不可用（", "). Offline alias matching was used instead.":"），已使用离线匹配。",
"Global ID ":"类别 ID ", " · Threshold ":" · 阈值 ", "Threshold ":"阈值 ",
"Seed **":"随机种子 **", "** · Best epoch **":"** · 最佳轮次 **", "Threshold **":"阈值 **",
" · action-interval clips · Valid evaluation":" · 动作切片 · 验证集评估",
"Valid-only one-vs-rest matrices · threshold ":"验证集一对其余混淆矩阵 · 阈值 ",
" selected on Valid · 31 separate 2×2 matrices, not a multiclass 31×31 matrix.":"，由验证集选定，共 31 张 2×2 矩阵。",
})

def t(value, language=None):
    language = language or st.session_state.get("ui_language", "en")
    if not isinstance(value, str) or language != "zh":
        return value
    if value in ZH:
        return ZH[value]
    for source in sorted(ZH, key=len, reverse=True):
        if len(source) > 8 and source in value:
            value = value.replace(source, ZH[source])
    return value

def display_format(function):
    language = st.session_state.get("ui_language", "en")
    return lambda value: str(t(function(value), language))

def display_data(value):
    if st.session_state.get("ui_language", "en") != "zh":
        return value
    if hasattr(value, "rename"):
        return value.rename(columns=t)
    return value


def option_formatter():
    language = st.session_state.get("ui_language", "en")
    return lambda value: str(t(value, language))

ZH.update({
"**Proposal · ResNet-18**":"**提案实验 · ResNet-18**",
"**AnimalAct 2.0 · VideoMAE Base**":"**AnimalAct 2.0 · VideoMAE 骨干**",
"1–16 unique real frames · full aspect ratio · no crop":"1–16 个真实帧 · 保持宽高比 · 完整画面",
})

ZH.update({
"Evaluation": "实验评估",
"AnimalAct 2.0 Evaluation": "AnimalAct 2.0 实验评估",
"Recognition method": "识别方法",
"Method progression: ResNet-18 → mean pooling → BiGRU → temporal attention; VideoMAE extends the encoder.": "方法演进：ResNet-18 → 平均池化 → BiGRU → 时间注意力；VideoMAE 扩展特征编码器。",
"Whole-video ResNet-18 uses 16 uniform samples and trains all 119 classes. Short videos may reuse frames.": "整视频 ResNet-18 均匀采样 16 帧，训练全部 119 类；短视频可能重复采样帧。",
"Experimental localization uses the frozen ResNet-18 + BiGRU attention features and has not beaten the whole-video baseline.": "实验性定位复用冻结的 ResNet-18 与 BiGRU 注意力特征，目前未超过整视频基线。",
"Whole-video classification": "整视频分类",
"Clip-based extensions": "切片方法扩展",
"Download classification evaluation": "下载分类评估结果",
"Whole-video experiment artifacts are unavailable in this installation.": "当前环境未提供整视频实验产物。",
"Different sample units, pretraining and training policies: these scores are separate experiments, not a controlled backbone ablation.": "样本单位、预训练与训练规则不同，各组成绩属于独立实验，不能视为仅更换骨干的消融对照。",
"Grounding Test scores are exploratory: Test was inspected across development rounds. The latest dense model was developed on Valid only.": "定位测试成绩为探索性结果：开发过程中曾多次查看测试结果。最新逐时间单元模型仅使用验证集开发。",
"From the proposed ResNet-18 baseline to temporal modeling and VideoMAE extensions.": "从提案中的 ResNet-18 基线出发，逐步加入时序建模与 VideoMAE 扩展。",
"VideoMAE · action-interval clips · Valid evaluation": "VideoMAE · 动作区间切片 · 验证集评估",
"ResNet-18 · mean": "ResNet-18 · 平均池化基线",
"ResNet-18 · bigru_mean": "ResNet-18 · BiGRU 平均池化",
"ResNet-18 · bigru_attention": "ResNet-18 · BiGRU 时间注意力",
"VideoMAE · bigru_mean": "VideoMAE · BiGRU 平均池化扩展",
"VideoMAE · bigru_attention": "VideoMAE · BiGRU 时间注意力扩展",
})

ZH.update({
"Actions and time intervals are detected automatically; no query is required.": "系统自动检测动作及时间区间，无需填写查询或选择动作。",
"Automatic time detection is not ready yet. Classification remains available; no timestamps will be invented.": "自动时间定位模型尚未就绪。仍可运行行为分类，当前不生成时间区间。",
"Classification summarizes the supplied video; automatic time detection is evaluated separately.": "分类概括输入视频中的动作；自动时间定位使用独立训练的时序检测头。",
"Detected actions and time intervals": "检测到的动作与时间区间",
"Experimental automatic detection. Scores are uncalibrated; sub-second boundary accuracy is not guaranteed.": "实验性自动检测，分数未经校准；当前不保证亚秒级边界精度。",
"No action interval passed the detector thresholds. This does not prove that no action occurred.": "没有动作区间达到检测阈值，这不代表视频中没有动作发生。",
"Start (seconds)": "起点（秒）", "End (seconds)": "终点（秒）",
"Duration (seconds)": "持续时间（秒）", "Match score": "匹配分数", "Time (seconds)": "时间（秒）",
"Download action intervals CSV": "下载动作时间区间 CSV",
"Experimental time detection is enabled. Results include candidate actions and intervals; errors remain frequent.": "已启用实验性时间定位。结果将显示候选动作及时间区间，仍存在较多错误。",
"Experimental preview: action labels and time intervals may be wrong. These are not verified annotations.": "实验性预览：动作类别和时间区间可能有误，不是经过核实的标注。",
})

ZH.update({'ResNet-18 · Weighted ensemble': 'ResNet-18 · 加权融合优化', 'BiGRU Mean 20% + Attention 80%': 'BiGRU 平均池化 20% + 注意力 80%', 'Weighted scores: 20% BiGRU Mean + 80% Attention. Weights and threshold selected on Valid; time detection remains a separate original model.': '融合分数：20% BiGRU 平均池化 + 80% 注意力。权重和阈值由验证集选择；时间区间检测仍使用原有独立模型。', 'Weighted ensemble optimization': '加权融合优化', 'Original models are retained. This module adds score fusion, Valid-only selection and comparison.': '保留原有模型，本模块新增分数融合、仅基于验证集的方案选择及对照分析。', 'Equal averaging has higher Test mAP, but the weighted policy was selected on Valid. No policy was reselected using Test.': '等权平均的测试 mAP 更高，但加权方案由验证集选定，未根据测试集重新选择方案。', 'F1 gains reflect fusion and threshold selection together. Exploratory results on a previously evaluated Test split; statistical significance is not established.': 'F1 提升来自融合与阈值选择的共同作用。这是既有测试集上的探索性结果，尚未证明统计显著性。', 'Download fusion experiment': '下载融合实验结果'})

ZH.update({})
