# AnimalAct 2.0 · COMP6116

当前动作区间级119类多标签识别系统，提供Video / Results / Evaluation三页Dashboard。**本仓库只包含AnimalAct 2.0，不包含已淘汰的AnimalAct 1.0、Model5/6旧方案、原始课程视频/标注或虚拟环境。**

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
- Secrets：**不需要**，不要上传GitHub访问令牌。

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

## 真实模型与数据流程

- 主力：冻结Kinetics VideoMAE + 单层BiGRU(hidden128/方向) + Class-specific Temporal Attention，Seed42、最佳Epoch21、固定Valid阈值0.12。
- 可选：相同冻结编码器 + BiGRU Mean，Seed42、最佳Epoch41、阈值0.07。
- 来源为Phase2.4，Phase2.5直接复用。deployment/model_manifest.json记录原Checkpoint SHA与导出头SHA；导出只移除了训练优化器等运行时不需要的状态，分类头参数逐Tensor不变。
- 视频按给定[start,end)窗口解码；<=16帧保留全部真实帧，否则均匀选16个唯一帧，不重复补帧。
- RGB完整画面、短边224等比PIL Resize、ImageNet归一化、不中心裁剪；空间Patch补零对齐、奇数帧增加一个内部零槽、矩形/变长位置编码插值，与已验证2.0适配器计算一致。
- 空间池化得到变长[Lt,768]，传入原Proposal头，输出119个独立sigmoid分数。可输出零个或多个动作；无训练正例的Defensive Rearing(ID23)只展示分数，禁止作为正常识别动作。
- 默认整文件作为一个Clip，亦可用户指定时间窗口。窗口边界是输入范围，**不是预测动作起止时间**。

## 完整研究资料、复现与继续开发

现已补齐独立研究包：[完整复现说明](research_tools/README.md)、[研究Release](https://github.com/0Ofan/Multi-Label-Animal-Action-Recognition/releases/tag/v2.0-research-reproduction)。约995MiB，含正式Clip Manifest及派生标签、来源索引、两套全量缓存、原Checkpoint与预测、校验/配置、当前2.0研究阶段资料和预训练权重。原始AnimalAct/视频、JSON标注和demo仍仅保留在课程成员本地；1.0实际代码/模型/缓存不包含。

```powershell
python research_tools/restore.py
python research_tools/run.py --workspace research_workspace --family videomae --model model3 --smoke
```

组员可恢复现有实验资料、重算Valid、按明确命令重新训练或在新目录改进。原固定阶段脚本和封存字节原样保留；为解决异机文件时间/路径、固定输出与配置限制，新增独立portable训练入口，复用原数学、模型和评价规则，不覆盖历史结果。恢复后有数据/代码/缓存内容SHA检查。

新增训练必须显式使用--train；上传工作本身未进行正式训练。加分项可基于代码及本地原始数据另行开发，但未提供已完成的Temporal Grounding实现。课程Brief与Proposal在[project_documents/](project_documents/)。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_portable -v
$env:RUN_MODEL_SMOKE='1'
.\.venv\Scripts\python.exe -m unittest tests.test_portable -v
```

第二种会执行实际预训练CPU编码器，对合成的3帧视频做接口冒烟测试，不宣称动物分类准确率。真实动物视频验收在发布前仅用本机Valid来源文件完成，不上传该视频、标签或Test数据。Linux CPU的GitHub Actions可手动运行，结果见Actions。

原阶段报告中的绝对路径、历史保护记录和当时运行指令属于历史证据，当前部署以本README为准。没有重新训练或改变任何阈值。
