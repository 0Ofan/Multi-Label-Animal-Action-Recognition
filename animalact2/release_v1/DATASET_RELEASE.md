# AnimalAct 2.0 正式数据清单 release_v1

本地元数据发布标识：`animalact2-2.0.0-release-v1`。清单包版本及Schema版本均为`1.0.0`。这是动作区间级多标签数据定义，不是AnimalAct 1.0的视频级训练结果，也不包含复制的视频、RGB缓存、模型特征或权重。原始视频仍在项目的`AnimalAct/`目录。

## 正式数量

| Split | 官方源视频 | 有发布Clip的父视频 | 原Clip ID | 排除原ID | 可用原ID | 合并减少 | 正式Clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2075 | 2063 | 4785 | 111 | 4674 | 7 | 4667 |
| Valid | 446 | 442 | 1009 | 29 | 980 | 1 | 979 |
| Test | 447 | 444 | 1042 | 19 | 1023 | 1 | 1022 |
| 合计 | 2968 | 2949 | 6836 | 159 | 6677 | 9 | 6668 |

6836个历史Clip ID全部有且仅有一个去向：6677个映射到6668个正式Clip，159个单独保留并排除。9组均为两个完全相同的有效时间窗口；合并前后的帧索引和标签一致，没有标签冲突。2968个父视频全部保留在父视频索引中，19个没有发布Clip的父视频也没有被删除或移出官方Split。

原始标注来源共有7123条，全部保留。不能将“排除一个以某标注为锚点的Clip”理解为删除该原始标注：该标注仍可能对其它合法窗口的多标签并集提供支持。

## 已批准规则

1. 固定官方Train/Valid/Test，保留P03等来源风险记录，不调整任何视频归属。
2. 时间窗为原始动作区间，采用半开区间`[start,end)`。以已核验实际帧数`N`及FPS计算`T=N/FPS`。`start>=end`保留异常；否则`start>=T`保留待复核；其它情况只在`end>T`时截断到T。原始JSON不修改。
3. 对相同父视频、相同有效开始和结束时间的已通过记录合并。不同时间窗即使恰好产生相同帧输入，也不额外合并。本版本没有新增原子切分、时间容差或重叠阈值。
4. 一个窗口的119维标签是所有动物、所有有效标注区间与该窗口有**严格正时间交集**的动作并集。每个正标签必须在该交集内存在真实帧支持。不把未标注时间作为全负背景，不借窗口外帧，不删除共标签。
5. 严格帧点满足`start <= k/FPS < end`，且`0 <= k < N`。1–3和4–15帧全部保留；16帧以上使用既有均匀采样，精确位置为`round(j*(n-1)/15), j=0..15`，round采用就近偶数。不会重复补齐16帧，不默认扩展上下文。
6. RGB完整画面，短边224等比Resize，PIL bilinear，不做中心裁剪或letterbox；沿用ImageNet均值`[0.485,0.456,0.406]`、标准差`[0.229,0.224,0.225]`。当前已核验视频对应单帧`[3,224,398]`。本阶段只发布预处理契约，没有重新读取像素或生成RGB。
7. 严格沿用Phase1.4.2的159条排除ID及原因。保留119个类别及其名称、全局ID；31个主类固定不变。Defensive Rearing（ID23）保留定义，当前没有可用训练正例。
8. Test只建立独立元数据清单，禁止用于训练、选模、阈值选择或调参。默认开发入口只列Train/Valid。

## 文件与路径

清单、索引里的路径分两种：Manifest等发布文件的路径相对于本目录；`video_path`、`annotation_path`及`annotation_ref`的路径部分相对于项目根目录。`dataset_index.json`的`project_root_relative_to_release`为`../..`。

| 文件 | 用途 |
| --- | --- |
| `dataset_index.json` | 三Split统一索引、路径基准、用途限制和数量 |
| `development_index.json` | 只包含Train/Valid的开发入口，不带Test标签 |
| `train/manifest.jsonl`、`valid/manifest.jsonl`、`test_metadata_only/manifest.jsonl` | 正式清单，三个Split物理分开 |
| 各分区`manifest.csv` | 对应JSONL的平表；列表/嵌套字段为JSON字符串 |
| `class_mapping.json` | 119类紧凑映射；不混用旧视频级类别支持数 |
| `main_evaluation_classes.json` | 固定31个主评估类及全局ID |
| `preprocessing.json` | 完整画面、时间采样、真实长度与mask契约 |
| `source_clip_index.csv` | 6836个历史Clip到新Clip或排除记录的唯一映射 |
| `annotation_source_index.csv` | 7123条原始标注UID、JSON Pointer、SHA及分区定位 |
| 各分区`annotation_sources.jsonl` | 完整原始标注字段及有效边界、帧支持；保留Phase1.4来源审计字段 |
| 各分区`label_source_links.csv` | 正标签到原始标注、精确交集的关系表 |
| `excluded_records.csv` | 159条排除记录的总索引及原因 |
| 各分区`excluded_records.jsonl`、`excluded_records.csv` | 排除记录与完整旧记录，不进入正式Manifest |
| `source_files_manifest.csv` | 2968个原始视频/JSON的位置、源SHA及核验依据 |
| `parent_video_index.csv` | 所有2968个官方父视频的发布与排除数量 |
| `merge_audit.csv` | 9组有效同窗合并及原Clip IDs |
| `statistics/` | Split、119类、固定31主类、帧长度、训练类别可用性统计 |
| `audit/` | 小样本一致性、等价输入、跨Split及既有来源风险、文件保护依据 |
| `data_validation.json` | 独立原JSON、密集帧点和标签复核结果 |
| `version.json`、`FILE_MANIFEST.csv`、`SHA256SUMS.txt` | 版本、文件清单、发布文件完整性校验 |
| `FINAL_VERIFICATION.json` | 封存时文件数、字节数、耗时和保护结果 |

`annotation_sources.jsonl`中保留的`simulation_only:true`说明该**来源审计记录**出自Phase1.4模拟结果；正式发布状态以`version.json`、`dataset_index.json`及正式Manifest为准。不能将来源表当作训练候选列表。

## 正式JSONL接口

- `clip_id`：带发布版本、Split、父视频与精确有效时间窗的稳定ID。`original_clip_ids`保留其全部历史身份。
- `parent_video_id`、`split`、`source_group_id`、视频及标注路径/SHA：继承官方元数据。
- `original_windows`：逐个原Clip保留`original_start_time`、`original_end_time`、`effective_start_time`、`effective_end_time`及对应精确分数、原Clip ID、标注UID/Pointer。合并可能有多个不同原终点，不能用单个顶层原终点覆盖它们。
- 顶层`effective_start_exact`、`effective_end_exact`是正式时间边界，优先用于精确计算；`start_seconds`、`end_seconds`是有效时间的浮点别名。
- `frame_indices`：按时间递增的唯一真实帧；`available_strict_frame_count`是区间内全部真实帧数；`effective_length`是实际输入帧数1–16，**不是秒数**。
- `valid_frame_mask`：与当前真实帧等长，全部True；未来若batch做数值padding，必须使用False屏蔽padding，不能重复视频帧。三种后续模型应共享同一套clip/frames/labels/length/mask；模型适配尚未在本阶段实施。
- `class_ids`及`target_119`：固定全局ID顺序的正类列表与119维二值target。
- `label_evidence`：逐正类保存所有贡献标注及精确交集、密集帧支持和采样帧支持。原数据没有提供独立的原始动作ID；`annotation_uid`由原JSON SHA及JSON Pointer稳定派生，不假冒原生ID。
- `source_annotation_uids/refs`：构成原窗口锚点的来源并集；`all_positive_label_source_uids`：全部正标签的贡献来源。两者含义不同。
- `sampled_label_coverage_warnings`：标签有密集支持但未被均匀采样命中的类别；保留原标签及原采样，不因此新增排除。

## 数据质量与使用边界

- 119类训练支持118类；Defensive Rearing为0。Valid/Test各有87类正标签；固定31主类在每个Split都存在可用支持。完整数量以`statistics/class_statistics_119.csv`及`main31_support.csv`为准。
- 1–3帧Clip：Train129、Valid33、Test29；4–15帧：Train925、Valid167、Test196；16帧以上：Train3613、Valid779、Test797。
- 采样覆盖警告是**Clip×类别对**：Train43、Valid9、Test11，总63，不是63个必然不同的视频。所有正式正标签均有密集帧点支持，但不保证均匀16帧看到每个短暂共标签。
- 不同时间窗、相同输入且标签相同的76组（Train57、Valid10、Test9；涉及155个Clip）按批准规则保留，详见`audit/equivalent_input_groups.csv`。正式有效同窗重复为0，保留输入中目标冲突为0。
- 跨Split的父视频ID、官方group、已知整视频SHA、源SHA+帧序列、源SHA+有效窗口身份键均未发现相同项。这不等于不同编码文件不存在相同画面或原始素材。
- 保留旧600组感知候选，其中244组跨Split，243组两端都有正式Clip。P03（Train JZLXSRAC / Valid TYAODRAC）已有较强同源证据，沿用官方归属，继续披露不确定性，不宣布“无泄漏”。本轮未重新进行像素或近重复搜索。
- Phase1.2的82个原Clip逐条对照：帧表及119维标签全部一致；78条发布、4条按既有规则排除。其中原待审9条里5条通过批准的终点截断恢复，不能将状态变化误报为“与旧状态完全一致”。

## 完整性与复核

独立校验器不导入构建函数，从2968份原JSON、官方Split重新构造有效区间，枚举真实帧网格并复算6668个Clip、12742个正标签组合、14127条来源关系；同时核对CSV/JSONL及统计、索引、合并与排除清单。没有读取视频像素、Test预测或模型。

原数据保护采用分层证据：488个历史阶段文件重新核对内容SHA，2983份构建时读取的来源文件核对SHA，其中含2968份原JSON；36124个旧项目文件检查大小/修改时间快照。模型及大视频未重新计算全量内容SHA，视频SHA沿用官方及Phase0已核验证据，不能称为本轮重新读取验证。Train/Valid的14个分区结果在进入Test元数据阶段前冻结，随后哈希不变。

`SHA256SUMS.txt`覆盖发布目录除其自身以外的所有文件，包含`FILE_MANIFEST.csv`及最终报告/验证记录。`FILE_MANIFEST.csv`列出除其自身和`SHA256SUMS.txt`以外的文件及字节数、SHA，避免自引用。校验的是**本清单包**；原视频/JSON的源SHA另列，不把外部文件误算入包内。

封存后可在项目根目录执行只读复核：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 animalact2\release_v1\verify_release_integrity.py
```

构建/完整数据验证脚本在发现封存文件时会拒绝重写。后续变更应发布新版本，不能覆盖本包。该发布为本地数据定义与元数据交付，不涉及上传或授权变更。

**Phase1.4.3完成后停止。未进入Phase1.5，未提取特征、下载模型、训练或开发前端。**
