# 最终整合版发布记录

版本：v2.1-final-integrated。来源为组员交付的AnimalAct_最新版.zip，原始ZIP及所需模型分别保存在同名GitHub Release中；原研究Release保留。

## 发布范围

- 保留Video / Results / Evaluation、双语、原有六种识别选项、119类结果、导出、历史实验与可选语义搜索。
- 默认完整视频ResNet-18分数融合：BiGRU Mean权重0.2，Attention权重0.8；Valid固定阈值0.20。
- 保留VideoMAE冻结编码器及原Seed42任务头；原片段采样和预处理不变。
- 既有实验性自动时间检测按原权重运行。时间检测异常时保留分类结果，不生成替代时间区间。
- 原始视频、JSON标注、密钥和虚拟环境不纳入公开发布；此次不读取数据集、不训练、不提取特征、不执行Test推理。

## 云端适配

最终包原先依赖本地权重文件。新增deployment/final_artifacts.py及final_model_manifest.json，从Release恢复15个路径对应的12份去重载荷，逐项检查SHA256，拒绝覆盖不匹配的既有Checkpoint。原VideoMAE资源继续使用原发布清单。Evaluation及模型信息不需要先下载权重。

Streamlit入口和仓库地址保持不变。main更新后，已有Community Cloud应用自动拉取并重建；如果应用休眠，打开原网址唤醒。首次模型识别需要下载约56.4MiB最终权重包；首次切换VideoMAE另需原212.2MiB包。Cloud使用CPU，不依赖本机运行。

## 验证依据

本地45项自动化检查全部通过，包括完整模型合成视频推理、双语三页面、Results、导出、掩码、真实时间检测及失败降级。合成视频仅用于接口验收，不能用于声称识别准确率。Linux CPU全新环境恢复模型并通过同样45项检查：[Actions验收](https://github.com/0Ofan/Multi-Label-Animal-Action-Recognition/actions/runs/45866697209)。

release-v1、ResNet第一轮、Phase2.4与Phase2.5的既有跟踪文件相对上一公开版本未变化。最终包的20份模型文件SHA256与原件一致，配置、结果、阈值不变；来源及适配文件清单见final_release_provenance.json。

## 使用与复现

```text
python -m pip install -r requirements.txt
python scripts/restore.py all
python -m streamlit run streamlit_app.py
```

Linux/macOS和Windows均使用项目虚拟环境运行以上命令。重新训练与原始数据复现入口仍见research_tools/README.md和proposal/COMPLETION.md；不会因启动网页而训练或评价Test。

Community Cloud现有应用应指向0Ofan/Multi-Label-Animal-Action-Recognition、main、streamlit_app.py，Python3.13。基础识别无需Secrets；可选LLM只用于动作名称检索。原始数据由成员本地保留。

## 线上最终验收

2026-11-27，原Streamlit网址已自动切换至本最终版，无需重新部署。使用独立无登录浏览器上传自制10秒合成视频，完成真实CPU融合分类及独立时间检测，Results正常显示，JSON含119类，CSV含119行；Evaluation读取已有实验成功。该测试只证明完整交互链路可用，不证明动作识别准确率。结构化记录见final_acceptance.json。

网站：<https://multi-label-animal-action-recognition-xh8tppjnmarxt9wuhk6wbx.streamlit.app/>。本地和Linux均未执行训练或Test推理。
