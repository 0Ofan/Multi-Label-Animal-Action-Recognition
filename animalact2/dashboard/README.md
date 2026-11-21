# AnimalAct 2.0 Dashboard portable edition

当前运行/部署/恢复说明以仓库根README.md为准。本目录复用已验收的2.0三个页面，默认使用同进程真实CPU推理，可选原本机HTTP模式。

旧工程中的frontend/inference依赖已经移除，不包含AnimalAct 1.0。原始AnimalAct文件夹不公开；派生Clip清单和缓存已按授权加入独立研究Release。网页推理只需要运行Release中的冻结VideoMAE和两个分类头。

部分历史技术测试需要原始课程数据和封存缓存；公开可执行测试为 `python -m unittest tests.test_portable -v`。原阶段报告是历史实验记录，不等于已部署云端成功的声明。
