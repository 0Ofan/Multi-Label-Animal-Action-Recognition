# AnimalAct · COMP6116 最终整合版


## 当前发布版本与网站入口

当前版本基于组员最终交付的 `AnimalAct_最新版.zip / AnimalAct_Integrated_Team`，基础发布号 **v2.1-final-integrated**，网页查询接入版本 **v2.1.1-bonus-interface**。默认模型是 **ResNet-18 + 20% BiGRU Mean + 80% Class-specific Temporal Attention分数融合**，整视频固定16帧、Seed6116、Valid阈值0.20。保留三个独立ResNet分类器，以及原VideoMAE BiGRU Mean和Attention两个可选模型，共六种识别方式。

网页仍使用 **Video / Results / Evaluation** 三页，支持中英文、上传预览、119类多标签分数、CSV/JSON导出和实验记录查看。实验性时间检测使用独立的4fps ResNet模型，错误仍较多；时间检测失败时保留分类结果并显示错误。

- 网站：<https://cnnproject-comp6116.streamlit.app/>
- 代码：本仓库 `main` 分支；入口 **`streamlit_app.py`**。
- Streamlit Community Cloud：已有应用继续使用同一仓库、`main`和上述入口，无需另建应用；代码推送后平台自动更新，依赖变化可能触发重建。需要手动恢复时，在Manage app中选择Reboot。
- 新建应用：在 <https://share.streamlit.io/> 选择本仓库、`main`、`streamlit_app.py`，Python **3.13**。基础识别不需要Secrets。
- 模型：最终ResNet/检测/研究权重在[v2.1-final-integrated Release](https://github.com/0Ofan/Multi-Label-Animal-Action-Recognition/releases/tag/v2.1-final-integrated)，原VideoMAE模型继续使用v2.0-dashboard Release。代码按需下载并校验SHA256，不把大权重放入Git提交。
- CPU：Community Cloud没有本项目的CUDA GPU；无需本机开机。上传视频在应用服务器计算，冷启动和长视频处理需要等待。
- 原始AnimalAct视频、原始标注及虚拟环境不公开；继续开发时使用组员本地合法持有的数据。历史研究Release及实验结果保留。

本地运行（Windows）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 可选：提前恢复全部推理模型；否则首次识别时按需恢复
.\.venv\Scripts\python.exe scripts/restore.py all
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Linux/macOS将Python路径换为`.venv/bin/python`。`scripts/restore.py final`只恢复最终ResNet与时间检测等权重；`runtime`只恢复VideoMAE。研究训练资料仍按下文`research_tools/`流程恢复，推理模型包不包含原始训练视频。

**评价口径**：新增整视频ResNet/融合结果和历史片段级ResNet/VideoMAE结果是不同实验。Evaluation分别显示已有记录，不重新计算Test预测，不把Valid结果写成Test结果。保留原阈值与权重，不进行新训练。


## 网页Bonus：按查询定位动作

已在Windows、Linux CPU环境通过53项检查，并在当前公网网站完成两条查询、区间导出、页面切换及分类结果隔离验收。记录见[bonus_acceptance.json](deployment/bonus_acceptance.json)。验收使用自制合成视频，不代表定位准确率。

在Video页上传视频后，向下找到 **Query-based temporal grounding (Bonus)** / **自然语言时序定位（Bonus）**：

1. 输入英文模板，例如 `A goat is walking.`。也可展开查询生成器，选择训练动物名称和动作，再点击“使用此查询”。
2. 点击“运行查询定位”。该操作独立于普通分类，使用已有ResNet-18 + BiGRU查询窗口评分器的第3轮最佳权重，CPU运行。
3. 查看最高分时间区间、备选窗口和时间轴，下载独立CSV或JSON；Results页也会保留本次查询结果。
4. 更换视频或修改查询后，旧查询结果失效，需要重新运行。切换Video/Results页面不会自动重复推理。

查询定位、无需查询的自动时间检测、动作名称筛选是三个不同功能。Results中的“筛选动作名称”只筛选119类分数，不会触发定位。Bonus模型不随上方分类模型选项切换，也不使用LLM。

模型存在长窗口偏置，不能可靠判断目标动作不存在；备选窗口不是多个独立行为事件，匹配分数不是校准概率。不在训练动物词表中的名称沿用原模型的未知动物嵌入，并显示提示。上传视频没有真实标注，因此不为它显示虚构IoU。历史IoU、mIoU和R@1仍在Evaluation的模板查询定位页签中展示，不重新评价Test。

可选本地HTTP接口：`GET /api/grounding`返回词表和模型来源；`POST /api/grounding?query=A%20goat%20is%20walking.&device=cpu`接收视频二进制正文及`X-Filename`文件名，返回独立查询定位结果。HTTP服务仍仅监听本机127.0.0.1；Streamlit Cloud使用同进程直接推理。

## 中文与英文界面

侧栏的 Language / 语言 可切换 English 和简体中文。中文入口使用 `http://127.0.0.1:8502/?lang=zh`，英文入口使用 `?lang=en`，统一实验评估页可追加 `&page=evaluation`。

两种语言共用同一套页面、推理、训练和评估代码。界面翻译集中在 `animalact2/dashboard/ui/i18n.py`。增加显示文本时使用 `t(...)` 并补充 ZH 翻译，选项的内部值和导出数据保留原始名称。运行 `python -m unittest tests.test_i18n -v` 检查三页两种语言与翻译条目，后续改动以此保持同步。

## Method progression and whole-video experiments

The Video, Results and Evaluation pages share the proposed method progression: ResNet-18 mean pooling, BiGRU mean pooling, temporal attention, then VideoMAE encoder extensions. Whole-video ResNet-18 is the default. Automatic action detection reuses dense ResNet features and predicts all action classes and boundaries without a text query; the existing template-query ranker is also available in the independent Bonus section on Video, alongside its retained evaluation records. Training, validation and test retain the official 2,075 / 446 / 447 video split. All 119 actions enter the new classification loss. See [proposal/COMPLETION.md](proposal/COMPLETION.md) for measured results and [proposal/results/results.json](proposal/results/results.json) for per-class counts. The historical interval-clip experiments below have their own evaluation unit.

From the project root, reproduce the whole-video experiments in an empty output directory:

```powershell
.\.venv\Scripts\python.exe -B -u -m proposal.experiment PATH_TO_AnimalAct.zip --mode full --frames 16 --epochs 40 --output proposal/results --weights proposal/resnet18-f37072fd.pth
.\.venv\Scripts\python.exe -B -u -m proposal.grounding PATH_TO_AnimalAct.zip --experiment proposal/results --output proposal/grounding_rank_results --epochs 40 --objective query_ranking
```

The first command fits the three existing classification heads on frozen ImageNet ResNet-18 features. Checkpoints and global thresholds are selected on Valid before any Test prediction. The second fits query-window scores using training interval IoU ranking targets and selects its checkpoint on Valid mIoU. Both commands refuse to overwrite completed results.

Re-evaluate a fixed classifier without fitting or selecting thresholds:

```powershell
.\.venv\Scripts\python.exe -B -m proposal.evaluate --model bigru_attention --split test --output proposal/recheck_attention_test.json
.\.venv\Scripts\python.exe -B -m unittest tests.test_proposal -v
```

The Video page accepts a complete video without a manual action or language query. Results displays automatic action intervals as a table and timeline, with interval CSV and full JSON exports. Classification and temporal detection have separate heads and scores. Before a full automatic detector is trained and reviewed, the interface explicitly reports that time detection is unavailable and retains classification only. Old query-conditioned weights cannot substitute for automatic detection.

To train the automatic detector using the existing read-only 4fps cache (Train/Valid only):

```powershell
.\.venv\Scripts\python.exe -B -u -m proposal.action_detection ../AnimalAct --mode train --cache proposal/boundary_results/features --output proposal/action_detection_results --epochs 40 --patience 8 --batch-size 8 --threads 1
```

The new head follows the existing ResNet-18 → BiGRU → per-class activity/start/end method. Same-action intervals from different animals are merged when overlapping; multiple actions can overlap. Validation includes one-to-one event precision/recall/F1 at temporal IoU 0.5 and at simultaneous ±0.5-second boundary tolerance, plus per-video/action mIoU and R@1 at 0.5/0.7. These action-only evaluation units differ from historical animal/action query units. Missing annotations are treated as background; precision still needs independent validation. Training uses in-memory cached features, batches of eight videos, one CPU thread and patience-eight early stopping; at most 40 epochs. It writes each epoch's metrics and protects prior output directories.

`results.json` must indicate a completed, non-smoke, reviewed run with `enabled_for_inference: true` before the website loads `proposal/action_detection_balanced_results/detector.pt`. Training leaves that flag false until result review. The balanced detector is now enabled for user-requested experimental preview, with explicit warnings; its measured precision is insufficient for verified localization. Smoke weights are rejected by inference even if accidentally enabled.


下文保留原AnimalAct 2.0动作区间级研究与复现说明；当前默认界面采用上文的最终整合方案。**本仓库只包含AnimalAct 2.0，不包含已淘汰的AnimalAct 1.0、Model5/6旧方案、原始课程视频/标注或虚拟环境。**

## 组员查看与验收

- [三种子实验报告](animalact2/phase2_5/FINAL_PHASE2_5_REPORT.md)
- [三种子汇总CSV](animalact2/phase2_5/summary_mean_std.csv)
- [九组实验明细](animalact2/phase2_5/all_seed_results.csv)
- [ResNet18第一轮](animalact2/first_round/FINAL_FIRST_ROUND_REPORT.md)
- [VideoMAE第一轮](animalact2/phase2_4/FINAL_PHASE2_4_REPORT.md)
- [正式数据定义](animalact2/release_v1/DATASET_RELEASE.md)
- [发布验收记录](DEPLOYMENT_ACCEPTANCE.md)

| VideoMAE分类头 | Valid mAP（3 seeds，mean ± sample std） | Valid Macro-F1 |
|---|---:|---:|
| Mean | 43.15 ± 0.57% |45.17 ± 0.83%|
| BiGRU Mean |46.28 ± 0.37%|46.53 ± 0.51%|
| BiGRU + Temporal Attention |46.78 ± 0.92%|46.20 ± 1.00%|

均为31主类、Clip级Valid指标，不是Test或外部视频准确率。强过拟合限制已经披露；页面不保证所有上传均正确。

## 无需本机运行：部署到Streamlit Community Cloud

登录 <https://share.streamlit.io/>，连接本GitHub仓库，选择Create app：

- Repository：`0Ofan/Multi-Label-Animal-Action-Recognition`
- Branch：`main`
- Main file path：`streamlit_app.py`
- Advanced settings → Python：`3.13`
- Secrets：基础识别**不需要**；仅当启用可选的 LLM 语义搜索时才需配置（见下文），不要上传GitHub访问令牌。

点击Deploy。部署成功后分享平台分配的 `https://....streamlit.app` 地址；该地址的实际开通需要仓库所有者在Streamlit登录并授权，不能仅靠Git push自动生成。不要把本机127.0.0.1链接当作公网网站。

该入口使用**同一Streamlit进程内的真实CPU推理**，不要求您的电脑开机，不启动独立8765 API，不依赖Windows路径或GPU/CUDA。初次识别会从本仓库Release下载约212MiB压缩模型包，解压并核验SHA256；后续复用已缓存的编码器与分类头。云实例休眠/重启后可能再次下载。Evaluation不会为显示表格而加载模型。

Cloud是Linux且有CPU/RAM资源限制，冷启动与推理可能比本机GPU慢；并发推理串行，上传上限200MiB。不是高并发商业托管。官方资源说明：<https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>。云端是否真正部署成功以Streamlit构建和运行日志为准，不能用本机测试代替。

## 本地下载运行

```powershell
git clone https://github.com/0Ofan/Multi-Label-Animal-Action-Recognition.git
cd Multi-Label-Animal-Action-Recognition
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Linux/macOS使用 `python3 -m venv .venv` 与 `.venv/bin/python`。建议Python3.13。requirements是**CPU跨平台运行配置**，不是原RTX5060训练环境；不应覆盖原工作区CUDA环境。如果使用保留的CUDA环境启动，也可由Video页选择CUDA。

初次推理会自动下载所需公开模型。也可提前下载：

```powershell
.\.venv\Scripts\python.exe scripts/restore.py runtime
# 可选：恢复2.0的24份原始Best/Last训练Checkpoint，用于实验核验
.\.venv\Scripts\python.exe scripts/restore.py checkpoints
```

发布资产与每文件SHA见deployment/model_manifest.json、checkpoint_archive.json。恢复脚本校验大小及SHA，并拒绝覆盖内容不同的Checkpoint。代码Git历史保持小；大模型使用GitHub Releases，不要求Git LFS配额。

## 保留的VideoMAE可选模型与数据流程

- VideoMAE Attention选项：冻结Kinetics VideoMAE + 单层BiGRU(hidden128/方向) + Class-specific Temporal Attention，Seed42、最佳Epoch21、固定Valid阈值0.12。
- VideoMAE BiGRU Mean选项：相同冻结编码器 + BiGRU Mean，Seed42、最佳Epoch41、阈值0.07。
- 来源为Phase2.4，Phase2.5直接复用。deployment/model_manifest.json记录原Checkpoint SHA与导出头SHA；导出只移除了训练优化器等运行时不需要的状态，分类头参数逐Tensor不变。
- 视频按给定[start,end)窗口解码；<=16帧保留全部真实帧，否则均匀选16个唯一帧，不重复补帧。
- RGB完整画面、短边224等比PIL Resize、ImageNet归一化、不中心裁剪；空间Patch补零对齐、奇数帧增加一个内部零槽、矩形/变长位置编码插值，与已验证2.0适配器计算一致。
- 空间池化得到变长[Lt,768]，传入原Proposal头，输出119个独立sigmoid分数。可输出零个或多个动作；无训练正例的Defensive Rearing(ID23)只展示分数，禁止作为正常识别动作。
- 默认整文件作为一个Clip，亦可用户指定时间窗口。窗口边界是输入范围，**不是预测动作起止时间**。

## 可选：LLM 动作语义搜索

Results 页支持用自然语言检索 119 个动作类（中英文均可），例如“攀爬或跳跃的动作”或 "aggressive behaviour"，在 Search mode 中切换 `Semantic (LLM)` / `Keyword`。

- LLM 只对表格和图表做筛选、重排与理由展示，**不改变**任何 sigmoid 分数、固定阈值或已预测标签；导出仍是全部 119 类原始分数。
- 只把**查询文本和动作类名**发送到配置的 OpenAI 兼容接口；上传的视频、帧和本机文件不会发送。
- 未配置密钥或接口调用失败时自动降级为本地双语别名/关键词匹配，云端部署仍可启动、识别和导出。
- 相同查询按“查询 + 模型 + 词表”缓存 1 小时，避免重复调用计费。

Streamlit Community Cloud：App settings → Secrets 添加：

```toml
[llm]
api_key = "sk-..."
base_url = "https://api.openai.com/v1"   # 任意 OpenAI 兼容端点
model = "gpt-4o-mini"
```

DeepSeek 示例：`base_url = "https://api.deepseek.com"`、`model = "deepseek-chat"`；通义千问兼容模式：`base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"`、`model = "qwen-plus"`。本机也可用环境变量 `ANIMALACT2_LLM_API_KEY`、`ANIMALACT2_LLM_BASE_URL`、`ANIMALACT2_LLM_MODEL`（优先级高于 Secrets）。参考 `.streamlit/secrets.toml.example`，不要把真实密钥提交到仓库。

## 完整研究资料、复现与继续开发

现已补齐独立研究包：[完整复现说明](research_tools/README.md)、[研究Release](https://github.com/0Ofan/Multi-Label-Animal-Action-Recognition/releases/tag/v2.0-research-reproduction)。约995MiB，含正式Clip Manifest及派生标签、来源索引、两套全量缓存、原Checkpoint与预测、校验/配置、当前2.0研究阶段资料和预训练权重。原始AnimalAct/视频、JSON标注和demo仍仅保留在课程成员本地；1.0实际代码/模型/缓存不包含。

```powershell
python research_tools/restore.py
python research_tools/run.py --workspace research_workspace --family videomae --model model3 --smoke
```

组员可恢复现有实验资料、重算Valid、按明确命令重新训练或在新目录改进。原固定阶段脚本和封存字节原样保留；为解决异机文件时间/路径、固定输出与配置限制，新增独立portable训练入口，复用原数学、模型和评价规则，不覆盖历史结果。恢复后有数据/代码/缓存内容SHA检查。

新增训练必须显式使用--train；上传工作本身未进行正式训练。加分项可基于代码及本地原始数据另行开发，历史研究包发布时未包含Temporal Grounding；最终整合版新增的相关研究代码与局限见`proposal/COMPLETION.md`，不可将其描述为可靠定位系统。课程Brief与Proposal在[project_documents/](project_documents/)。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_portable -v
$env:RUN_MODEL_SMOKE='1'
.\.venv\Scripts\python.exe -m unittest tests.test_portable -v
```

第二种会执行实际预训练CPU编码器，对合成的3帧视频做接口冒烟测试，不宣称动物分类准确率。真实动物视频验收在发布前仅用本机Valid来源文件完成，不上传该视频、标签或Test数据。Linux CPU的GitHub Actions可手动运行，结果见Actions。

原阶段报告中的绝对路径、历史保护记录和当时运行指令属于历史证据，当前部署以本README为准。没有重新训练或改变任何阈值。
