# 植物幼苗图像分类（机器学习课程设计 · 任务一）

基于 ImageNet 迁移学习的五分类图像识别，面向极小样本、细粒度场景。
训练集仅 500 张（五类各 100 张），测试集 378 张，Kaggle 排行榜最佳成绩 **0.9312**。

## 任务与数据

| 项目 | 内容 |
|---|---|
| 类别 | Black-grass、Common wheat、Loose Silky-bent、Scentless Mayweed、Sugar beet |
| 训练集 | 500 张，五类各 100 张 |
| 测试集 | 378 张，无标注 |
| 评价 | 分类准确率（排行榜） |
| 提交格式 | 两列 CSV：`ID,Category` |

> **数据不包含在本仓库中。** 原始数据来自课程/Kaggle 比赛，按比赛规则不便再分发。
> 复现时请自行下载，并放到 `data/dataset-for-task2/dataset-for-task2/{train,test}`。

## 方法概要

```
任意尺寸图像 → 补边成正方形 → 384×384 uint8 缓存
      ├─ 训练：随机裁剪缩放/翻转/±180°旋转/颜色抖动/随机擦除
      └─ 推理：确定性缩放 + 8 视角 TTA
              ↓
   ImageNet 预训练骨干（EfficientNet-B0 / ResNet50 / ConvNeXt-Tiny / ResNet18）
   替换分类头为 Linear(d, 5)
              ↓
   判别式学习率（骨干 1e-4 / 头 1e-3）+ OneCycleLR + 标签平滑 0.1
              ↓
   五折分层交叉验证 × 4 骨干 = 20 个模型
              ↓
   先按折平均 → 再按骨干平均 → 决策后处理
              ↓
        submission.csv
```

### 关键设计取舍

| 设计 | 理由 |
|---|---|
| ImageNet 迁移学习 | 每类仅 100 张；从零训练（400 万参数 vs 500 样本）无法泛化 |
| 五折分层交叉验证 | 单一 100 张验证集标准误约 ±4 个百分点，无法分辨 1~2 点的改进；五折降至 ±2 点并附带 5 个模型 |
| 判别式学习率 | 骨干已属较好解需微调，分类头随机初始化需快速收敛；单一学习率二者不可兼得 |
| 四骨干集成 | 集成收益取决于**误差去相关**：同族骨干错误高度相关，平均无效 |
| 8 视角 TTA | 正俯视拍摄下翻转/旋转不改变类别；在 softmax 空间平均（概率才可加） |
| 放弃 Mixup | 本任务类间差异恰为细微叶片形态，混合样本会破坏判别线索 |

## 实验结果

四骨干 × 五折集成，OOF（每张图由没见过它的模型预测）：

| 指标 | 数值 |
|---|---|
| 准确率 / 平衡准确率 | 0.9260 / 0.9260 |
| macro-F1 | 0.9256 |

| 类别 | 召回率 | 精确率 | F1 |
|---|---|---|---|
| Black-grass | 0.830 | 0.814 | 0.822 |
| Common wheat | 0.970 | 0.942 | 0.956 |
| Loose Silky-bent | 0.830 | 0.883 | 0.856 |
| Scentless Mayweed | 1.000 | 1.000 | 1.000 |
| Sugar beet | 1.000 | 0.990 | 0.995 |

误差集中在 Black-grass 与 Loose Silky-bent 这一混淆对上（两者均为禾本科，幼苗期形态接近）。

## 一个值得记录的现象：类别先验偏置

训练集五类严格均衡，故测试集真实分布应接近每类 20%，但模型在测试集上的原始预测为：

| 类别 | 训练集先验 | 模型预测 |
|---|---|---|
| Black-grass | 20.0% | 18.8% |
| **Common wheat** | 20.0% | **8.7%** |
| Loose Silky-bent | 20.0% | 26.7% |
| Scentless Mayweed | 20.0% | 23.8% |
| Sugar beet | 20.0% | 22.0% |

该偏置源于 ImageNet 标签先验的继承（禾本科杂草样本远多于栽培作物），表现为**决策边界整体偏移**，
在 macro-F1 一类宏平均指标上被显著放大，且无法靠增加轮次或更换骨干消除。

仓库中实现了两种不使用标签的修正，并对比了其适用边界：

- **OOF 决策权重修正**（`tune_decision_weights`）：在折外概率上学每类乘数，附留一折检验。
  OOF 上 macro-F1 0.9256 → 0.9391，但留一折净增益仅 +0.0095。
- **直推式先验修正**（`prior_fix`）：直接测量待预测集合的预测分布并推回均匀先验。

**实测结论**：将 OOF 学到的权重施于测试集后，分布反而更偏（平均偏离度 5.00 → 6.38）。
原因是集成后的 OOF 分布本身已接近均匀，已无偏置可修，优化器只能牺牲难分类别来抠分；
而测试集偏置远大于验证集，方向恰好相反。**修正工具必须匹配偏置的来源。**

## 复现

### 环境

```bash
pip install -r requirements.txt      # 仅 torch / torchvision / numpy / Pillow
```

不需要 sklearn、pandas、scipy —— 分层 K 折、准确率、混淆矩阵、macro-F1 均为自行实现。

### Kaggle（推荐，有 GPU）

`File → Import Notebook` 上传 `notebooks/plant_classification_kaggle.ipynb`，按顺序运行即可。
配置单元格默认四骨干 × 5 折 × 25 轮，GPU 上约 54 分钟。

### 命令行

```bash
python src/plant_classifier.py cache                                  # 建立图像缓存
python src/plant_classifier.py train --model efficientnet_b0 resnet50 convnext_tiny resnet18
python src/plant_classifier.py predict --model efficientnet_b0 resnet50 convnext_tiny resnet18 \
       --prior-strength 0.5 --out submission.csv
```

### 验证

```bash
python tools/verify_all.py --fast     # notebook 一致性 + 单元测试，不需训练
python tools/verify_all.py            # 追加 1 轮端到端（缓存→训练→TTA→提交）
```

## 仓库结构

```
.
├─ src/plant_classifier.py                # 全部训练/预测逻辑（单文件，无第三方依赖）
├─ notebooks/
│   └─ plant_classification_kaggle.ipynb  # Kaggle 可直接运行的 Notebook
├─ tools/
│   ├─ verify_all.py                      # 一键验证
│   ├─ test_pipeline.py                   # 单元测试（分层K折/TTA/指标/决策权重/OOF组装）
│   ├─ verify_end2end.py                  # 端到端验证
│   ├─ build_notebook.py                  # 由 src 生成 notebook
│   ├─ rebuild_notebook.py                # 只同步内嵌源码
│   ├─ validate_notebook.py               # 校验 notebook 与 src 一致
│   ├─ run_matrix.py                      # 批量训练 (模型 × 折)
│   ├─ auto_pipeline.py                   # 训练结束后自动预测 + 写提交
│   └─ report.py                          # 生成实验报告
├─ report/
│   ├─ 任务一_植物分类_实验报告.docx         # 实验报告（精简版）
│   ├─ build_report_compact.py            # 报告生成脚本
│   └─ figures/                           # 报告插图
├─ examples/submission_example.csv        # 提交文件格式示例
├─ requirements.txt
└─ README.md
```

`tools/build_notebook.py` 会把 `src/plant_classifier.py` 的原文嵌入 notebook，
因此 Notebook 与命令行跑的是同一份代码；`validate_notebook.py` 可校验二者逐字节一致。

## 说明

- 本仓库只包含代码与报告，**不含数据集、训练缓存与模型权重**（见 `.gitignore`）。
- `examples/submission_example.csv` 由本地单折模型生成，仅用于演示提交格式，
  不代表排行榜成绩；正式成绩请以 Kaggle 提交为准。

## 许可

课程作业代码，仅供学习与复现参考。
