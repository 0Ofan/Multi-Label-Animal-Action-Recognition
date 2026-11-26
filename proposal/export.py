"""Export report-ready whole-video metrics and deterministic unseen-video examples."""

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .experiment import save_json


def main():
    root = Path(__file__).resolve().parent
    result = json.loads((root / "results/results.json").read_text())
    for model, record in result["models"].items():
        with (root / "results" / f"{model}_test_per_class.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=["class_id", "action", "AP", "F1", "positive_videos", "TN", "FP", "FN", "TP"])
            writer.writeheader()
            for row in record["test"]["per_class"]:
                (tn, fp), (fn, tp) = row["confusion_2x2_tn_fp_fn_tp"]
                writer.writerow(dict(class_id=row["class_id"], action=row["action"], AP=row["ap"],
                                     F1=row["f1"], positive_videos=row["positive_videos"],
                                     TN=tn, FP=fp, FN=fn, TP=tp))
    lines = [
        "# 提案实验完成记录",
        "",
        "这次实验按已提交提案的整视频口径执行。训练 2,075 个、验证 446 个、测试 447 个视频，全部 119 类进入 BCE 损失，主要指标按固定 31 类计算。",
        "",
        f"三组复用原项目的 ResNet-18 分类头。冻结 ImageNet 预训练骨干，每个视频均匀采样 {result['frames']} 帧。随机种子为 {result['seed']}，最多训练 {result['epochs']} 轮，以验证集 mAP 选检查点，连续 8 轮未提升则早停。全局阈值由验证集 Macro-F1 选择，全部选择完成后计算 Test。",
        "",
        "## 分类与并发动作",
        "",
    ]
    for model, record in result["models"].items():
        metric = record["test"]
        concurrent = metric["concurrent_subset"]
        lines.append(
            f"- **{model}**，第 {record['selected_epoch']} 轮，阈值 {record['threshold']:.2f}。"
            f"Valid mAP {record['validation']['map']:.2%}。Test mAP {metric['map']:.2%}，"
            f"Macro-F1 {metric['macro_f1']:.2%}。108 个并发视频的动作召回率 "
            f"{concurrent['recall']:.2%}，主类预测精确率 {concurrent['precision']:.2%}。"
        )
    ground = json.loads((root / "grounding_rank_results/results.json").read_text())
    metric = ground["test"]["all"]
    selective = ground["test"]["temporally_selective"]
    lines += [
        "",
        "## 模板自然语言时序定位",
        "",
        f"复用按 Valid 选定的 {ground['classification_kind']} 的 ResNet-18 与 BiGRU，冻结视频特征，"
        "训练动物嵌入、动作嵌入和候选窗口评分头。查询格式为 A {animal} is {action}.，"
        "候选窗口为 16 个均匀时间单元的所有连续组合。根据最大时间 IoU 的四次方构建查询内的窗口排序目标。"
        "定位检查点按 Valid mIoU 选择。模型接收固定词表的动物和动作，未知动物使用零嵌入。",
        "",
        f"测试集共 {metric['queries']} 条正查询，mIoU {metric['miou']:.2%}，"
        f"R@1@0.5 {metric['r1_iou_05']:.2%}，R@1@0.7 {metric['r1_iou_07']:.2%}。"
        f"直接返回整视频的基线 mIoU 为 {metric['whole_video_baseline_miou']:.2%}。",
        "当前学习到的定位分数仍偏向较长窗口，整体效果低于整视频基线，应作为有明确局限的定位原型陈述。",
        "",
        f"整视频最佳 IoU 小于 0.5 的时间选择性子集共 {selective['queries']} 条，"
        f"定位 mIoU {selective['miou']:.2%}，R@1@0.5 {selective['r1_iou_05']:.2%}，"
        f"R@1@0.7 {selective['r1_iou_07']:.2%}。该子集帮助检查短动作的定位能力。",
        "",
        "评估只取最高分预测，命中任一真实区间即为命中。界面显示的后续区间是去重后的候选，"
        "其评分没有校准为概率。此版本支持模板查询。自然措辞泛化、无目标查询拒识和多事件完整召回尚需独立评估。",
        "",
        "## 复现与边界",
        "",
        "新增代码位于 proposal，历史切片版结果继续由 Evaluation 页面展示，整视频结果由 Proposal 页面展示。"
        "两个版本的样本单位、预训练来源和训练配置不同，成绩应分别陈述。",
        "",
        "特征缓存、模型权重、检查点选择记录、逐视频预测和逐类混淆计数都保存在 proposal/results。"
        "定位逐查询参考区间、预测区间和 IoU 保存在 proposal/grounding_rank_results/results.json。"
        "本次为一个固定种子的完整实验，固定划分本身仍存在既有同源素材风险。",
        "",
        "四名成员的实际贡献及反思需依据真实工作记录填写。已有方案分配不能代替实际贡献证据。",
        "",
        "定位开发共完成两轮。首轮 IoU 软标签 BCE 在 Valid 上退化为整视频预测；"
        "第二轮依据该 Valid 现象改用查询内窗口排序，两个版本均保留。两轮都分别计算了 Test，"
        "应披露这种开发顺序，Test 分数仅作探索性结果，不能声称为一次盲测。"
        "分类三组的检查点和阈值均在首次分类 Test 推理前固定。",
        "",
        "CUDA 原环境的固定模型复算记录与 CPU 跨版本复算记录一并保留。"
        "PyTorch 2.14 CPU 与 2.5.1 CUDA 的 GRU 分数有小幅数值差异，靠近阈值的标签可能变化。",
        "",
    ]
    (root / "COMPLETION.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "COMPLETION.md")


if __name__ == "__main__":
    main()
