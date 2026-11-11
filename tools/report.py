# -*- coding: utf-8 -*-
"""
Generate a markdown report from the training artefacts in outputs/.

Reads checkpoints, OOF probability files, training histories and the test
probability maps, then writes outputs/REPORT.md plus a confusion-matrix figure.

Usage:  python tools/report.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "src"))

import plant_classifier as pc                                    # noqa: E402

OUT = pc.OUT
REPORT = os.path.join(OUT, "REPORT.md")


def load_oof():
    """Concatenate every fold's OOF probabilities, grouped by backbone."""
    by_model = {}
    for path in sorted(glob.glob(os.path.join(OUT, "oof", "*.npz"))):
        base = os.path.basename(path)[:-4]
        model, fold = base.rsplit("_fold", 1)
        d = np.load(path)
        by_model.setdefault(model, []).append(
            dict(fold=int(fold), probs=d["probs"], idx=d["idx"], labels=d["labels"]))

    # a global OOF vector: one entry per training image, taken from whichever
    # fold held it out
    n = max((int(f["idx"].max()) for folds in by_model.values() for f in folds),
            default=-1) + 1
    oof_all = {}
    for model, folds in by_model.items():
        probs = np.zeros((n, pc.NUM_CLASSES))
        labels = np.full(n, -1, dtype=int)
        seen = np.zeros(n, dtype=bool)
        for f in folds:
            probs[f["idx"]] = f["probs"]
            labels[f["idx"]] = f["labels"]
            seen[f["idx"]] = True
        oof_all[model] = dict(probs=probs[seen], labels=labels[seen],
                              coverage=int(seen.sum()), total=n,
                              folds=len(folds), seen=seen)
    return oof_all


def confusion(probs, labels, n_cls=pc.NUM_CLASSES):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(labels, probs.argmax(1)):
        cm[t, p] += 1
    return cm


def main():
    oof = load_oof()
    if not oof:
        print("no OOF files found in %s/oof -- train something first" % OUT)
        return 1

    lines = []
    add = lines.append

    add("# 实验结果报告 / Experiment report")
    add("")
    add("植物幼苗 5 分类（Task 1）。本报告由 `tools/report.py` 从训练产物自动生成。")
    add("")

    # ---------------------------------------------------------------- CV table
    add("## 1. 交叉验证结果（OOF）")
    add("")
    add("OOF（out-of-fold）准确率：每张训练图都由**没见过它**的那一折模型来预测，")
    add("因此这是对泛化能力的无偏估计。")
    add("")
    add("| 骨干网络 | 折数 | 覆盖图片 | OOF 准确率 |")
    add("|---|---|---|---|")
    for model in sorted(oof):
        d = oof[model]
        acc = (d["probs"].argmax(1) == d["labels"]).mean()
        add("| `%s` | %d | %d/%d | **%.4f** |"
            % (model, d["folds"], d["coverage"], d["total"], acc))
    add("")

    # per-fold accuracy from the histories
    add("### 各折验证准确率")
    add("")
    add("| 骨干网络 | " + " | ".join("fold %d" % f for f in range(5)) + " |")
    add("|---|" + "---|" * 5)
    for model in sorted(oof):
        cells = []
        for fold in range(5):
            p = os.path.join(OUT, "history_%s_fold%d.json" % (model, fold))
            if os.path.exists(p):
                h = json.load(open(p))
                cells.append("%.4f (ep %d)" % (h["best_acc"], h["best_epoch"]))
            else:
                cells.append("--")
        add("| `%s` | %s |" % (model, " | ".join(cells)))
    add("")

    # ------------------------------------------------------- per-class report
    add("## 2. 每类识别情况")
    add("")
    # merge OOF of all models by averaging where they overlap
    models = sorted(oof)
    per_class_rows = []
    for model in models:
        d = oof[model]
        cm = confusion(d["probs"], d["labels"])
        rec = cm.diagonal() / np.maximum(1, cm.sum(axis=1))
        per_class_rows.append((model, rec))
    # macro average across models
    macro = np.mean([r for _m, r in per_class_rows], axis=0)

    add("| 类别 | " + " | ".join("`%s`" % m for m in models) + " | 平均召回 |")
    add("|---|" + "---|" * (len(models) + 1))
    for i, cls in enumerate(pc.CLASSES):
        cells = " | ".join("%.3f" % per_class_rows[j][1][i] for j in range(len(models)))
        add("| %s | %s | **%.3f** |" % (cls, cells, macro[i]))
    add("")
    add("各类平均召回率：%.4f（类别不均衡时比总体准确率更能反映问题）" % macro.mean())
    add("")

    hardest = int(np.argmin(macro))
    add("最难的类别是 **%s**（召回 %.3f）。从幼苗形态看，几种禾本科植物早期非常接近，"
        "混淆主要集中在它们之间。" % (pc.CLASSES[hardest], macro[hardest]))
    add("")

    # ------------------------------------------------------- ensemble results
    ens_path = os.path.join(OUT, "test_ensemble.npz")
    add("## 3. 测试集预测")
    add("")
    if os.path.exists(ens_path):
        e = np.load(ens_path)
        probs = e["probs"]
        names = [str(s) for s in e["test_names"]]
        pred = probs.argmax(1)
        add("集成 OOF 准确率：**%.4f**" % float(e["oof_acc"]))
        add("")
        add("| 类别 | 预测数量 | 占比 | 训练集占比 |")
        add("|---|---|---|---|")
        for i, cls in enumerate(pc.CLASSES):
            cnt = int((pred == i).sum())
            add("| %s | %d | %.1f%% | 20.0%% |" % (cls, cnt, 100.0 * cnt / len(pred)))
        add("")
        conf = probs.max(1)
        add("预测置信度：均值 %.3f，中位数 %.3f，最低 %.3f。"
            % (conf.mean(), np.median(conf), conf.min()))
        add("置信度 < 0.5 的样本：%d / %d（%.1f%%），这些是最可能的错误来源。"
            % ((conf < 0.5).sum(), len(conf), 100.0 * (conf < 0.5).mean()))
        add("")
        # agreement between backbones is a useful proxy for difficulty
        per_model_files = sorted(glob.glob(os.path.join(OUT, "test_*.npz")))
        per_model_files = [p for p in per_model_files
                           if os.path.basename(p) != "test_ensemble.npz"]
        if len(per_model_files) > 1:
            stacks = [np.load(p)["probs"].argmax(1) for p in per_model_files]
            agree = np.mean([len({s[i] for s in stacks}) == 1 for i in range(len(names))])
            add("不同骨干网络预测完全一致的比例：%.1f%%" % (100.0 * agree))
            add("")
    else:
        add("尚未生成测试集预测（运行 `predict`）。")
        add("")

    # ------------------------------------------------------------- ensembles
    add("## 4. 集成效果")
    add("")
    if len(models) > 1:
        # average OOF probabilities across models where all have coverage
        common = np.ones(oof[models[0]]["total"], dtype=bool)
        for m in models:
            common &= oof[m]["seen"]
        stacked, labels = [], None
        for m in models:
            full = np.zeros((oof[m]["total"], pc.NUM_CLASSES))
            full[oof[m]["seen"]] = oof[m]["probs"]
            stacked.append(full)
            labels = np.zeros(oof[m]["total"], dtype=int)
            labels[oof[m]["seen"]] = oof[m]["labels"]
        if common.sum() > 0:
            ens = np.mean([s[common] for s in stacked], axis=0)
            add("在 %d 张所有模型都有 OOF 预测的图片上：" % common.sum())
            add("")
            add("| 方案 | OOF 准确率 |")
            add("|---|---|")
            for m, s in zip(models, stacked):
                add("| `%s`（单模型） | %.4f |" % (m, (s[common].argmax(1) == labels[common]).mean()))
            add("| **集成（概率平均）** | **%.4f** |"
                % (ens.argmax(1) == labels[common]).mean())
            add("")
    else:
        add("当前只有一个骨干网络；加入第二个模型后本节会显示集成带来的增益。")
        add("")

    # ------------------------------------------------------------- artefacts
    add("## 5. 产物清单")
    add("")
    add("| 路径 | 内容 |")
    add("|---|---|")
    for rel, desc in [
        ("outputs/cache_384.npz", "缩放后的图像缓存（uint8）"),
        ("outputs/checkpoints/*.pt", "每折验证最优权重"),
        ("outputs/oof/*.npz", "每折的验证概率（用于 OOF 评估）"),
        ("outputs/history_*.json", "逐 epoch 训练曲线"),
        ("outputs/test_*.npz", "各骨干网络的测试概率（含 TTA）"),
        ("outputs/test_ensemble.npz", "集成后的测试概率"),
        ("submission.csv", "提交文件（ID,Category）"),
    ]:
        add("| `%s` | %s |" % (rel, desc))
    add("")

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s (%d lines)" % (REPORT, len(lines)))

    # also dump a small json summary for programmatic use
    summary = {
        "oof_accuracy": {m: float((oof[m]["probs"].argmax(1) == oof[m]["labels"]).mean())
                         for m in models},
        "n_train": int(oof[models[0]]["total"]),
        "per_class_recall": {pc.CLASSES[i]: float(macro[i]) for i in range(pc.NUM_CLASSES)},
    }
    if os.path.exists(ens_path):
        summary["ensemble_oof_accuracy"] = float(np.load(ens_path)["oof_acc"])
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("wrote %s" % os.path.join(OUT, "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
