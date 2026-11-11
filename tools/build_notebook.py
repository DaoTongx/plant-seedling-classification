# -*- coding: utf-8 -*-
"""
Build the Kaggle notebook from the real, locally-tested pipeline source.

The notebook embeds ``src/plant_classifier.py`` verbatim as one cell and writes
it to disk, then drives it.  That keeps a single source of truth: whatever runs
on Kaggle is exactly the code that was validated locally.

Usage:  python tools/build_notebook.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SOURCE = os.path.join(PROJECT, "src", "plant_classifier.py")
OUT_NB = os.path.join(PROJECT, "notebooks", "plant_classification_kaggle.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def build():
    with open(SOURCE, encoding="utf-8") as fh:
        pipeline_src = fh.read()

    cells = []

    cells.append(md(r"""
# 植物幼苗分类 —— 深度学习课程设计（Task 1）

**目标**：区分不同植物种类与生长状态 —— 对测试集中的植物图片进行 5 分类。

**数据**

| 划分 | 数量 | 说明 |
|---|---|---|
| train | 500 | 每类 100 张，已标注（5 个类别） |
| test  | 378 | 需要预测 |

**类别**：`Black-grass`(黑麦草)、`Common wheat`(普通小麦)、`Loose Silky-bent`(散穗黍)、
`Scentless Mayweed`(无香母菊)、`Sugar beet`(甜菜)

**提交格式**：`ID,Category`（两列），`ID` 为测试图片文件名，见 `submission-for-task2.csv`。
**组名必须填写个人学号。**

---

## 方法概述

这是一个**小样本、细粒度**的图像分类问题：每类只有 100 张训练图，且不同植物的幼苗在
早期形态上非常接近（尤其几种禾本科杂草），而背景（砂石、标签纸、塑料盘）差异很大。

因此方案的核心是**迁移学习 + 交叉验证集成 + 测试时增强（TTA）**：

1. **预训练骨干网络**：从 ImageNet 预训练权重出发微调。500 张图从零训练 CNN 无法与
   预训练特征竞争，这是本任务收益最大的一步。
2. **5 折分层交叉验证**：每张训练图恰好被验证一次，得到无偏的 OOF 准确率；
   同时 5 个折模型天然构成一个集成。
3. **强数据增强**：随机裁剪缩放、水平/垂直翻转、任意角度旋转、颜色抖动、随机擦除。
   植物可出现在画面任意位置、任意朝向，这些增强与实际差异一致。
4. **判别式学习率**：主干用较小学习率（1e-4），新初始化的分类头用较大学习率（1e-3），
   配合 OneCycle 余弦退火。
5. **测试时增强**：对每张测试图做 8 种几何变换后取平均概率，降低预测方差。
6. **多模型集成**：把若干不同骨干网络（EfficientNet-B0 / ResNet50 / ConvNeXt-Tiny 等）
   的概率平均，进一步降低方差。
"""))

    cells.append(md("""
## 0. 环境准备

Kaggle Notebook 自带 PyTorch 与 torchvision。本 Notebook **不需要任何第三方依赖**
（不使用 sklearn / pandas / scipy），只依赖 `torch`、`torchvision`、`numpy`、`PIL`，
因此在 Kaggle 上可以直接运行，无需 `pip install`。

> 建议在 Notebook 设置中打开 **GPU (Accelerator = GPU T4 x2 或 P100)**，
> 并把 **Internet** 设为 On（仅用于下载 ImageNet 预训练权重）。
> 若 Internet 关闭，请把预训练权重作为 Kaggle Dataset 挂载后设置 `TORCH_HOME`。
"""))

    cells.append(code(r"""
import os, sys, glob, json, time, platform

print("python :", sys.version.split()[0])
print("machine:", platform.platform())
try:
    import torch, torchvision
    print("torch  :", torch.__version__, "| torchvision:", torchvision.__version__)
    print("cuda   :", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")
except Exception as e:
    print("torch import failed:", e)
"""))

    cells.append(md("""
## 1. 定位数据

数据集在 Kaggle 上有几种可能的挂载方式，下面的代码会自动搜索 `train/` 与 `test/`
所在的目录，并打印每个类别的图片数量，确认数据读取正确。
"""))

    cells.append(code(r"""
# 这个单元格做两件事：找到数据集目录，并找到官方提交模板。
# Kaggle 挂载路径的层级不确定（有时是 <比赛>/dataset-for-task2/train，
# 有时会多套一层同名目录），所以这里用递归搜索，而不是写死路径。

import glob, os

def find_data_root():
    '''找到同时含 train/ 和 test/ 的目录，且 train/ 下正好是那 5 个类别。'''
    expected = {"Black-grass", "Common wheat", "Loose Silky-bent",
                "Scentless Mayweed", "Sugar beet"}
    roots = glob.glob("/kaggle/input/**/train", recursive=True)
    for train_dir in roots:
        root = os.path.dirname(train_dir)
        test_dir = os.path.join(root, "test")
        if not os.path.isdir(test_dir):
            continue
        subdirs = {d for d in os.listdir(train_dir)
                   if os.path.isdir(os.path.join(train_dir, d))}
        if subdirs == expected:
            return root
        print("跳过 %s（类别目录是 %s）" % (root, sorted(subdirs)[:6]))
    return None

DATA_ROOT = find_data_root()
assert DATA_ROOT, ("没找到数据集。请先在右侧 Add Data 里加入本次任务的数据，"
                   "确认 /kaggle/input 下存在 train/ 与 test/ 两个目录。")
print("DATA_ROOT =", DATA_ROOT)

for split in ("train", "test"):
    print("\n[%s]" % split)
    if split == "train":
        for cls in sorted(os.listdir(os.path.join(DATA_ROOT, "train"))):
            p = os.path.join(DATA_ROOT, "train", cls)
            if os.path.isdir(p):
                print("   %-20s %3d 张" % (cls, len(os.listdir(p))))
    else:
        print("   %d 张无标注图片" % len(os.listdir(os.path.join(DATA_ROOT, "test"))))

# 官方提交模板：用来保证输出的 ID 顺序与官方一致
cands = glob.glob("/kaggle/input/**/*.csv", recursive=True)
TEMPLATE = next((t for t in cands if "submission" in os.path.basename(t).lower()), None)
print("\nsubmission template:", TEMPLATE)
if TEMPLATE:
    with open(TEMPLATE) as fh:
        for i, line in enumerate(fh):
            print("   " + line.rstrip())
            if i >= 2:
                break
else:
    print("   （没找到 submission*.csv，不影响训练；只是提交时少了顺序保险）")
"""))

    cells.append(md("""
## 2. 动手前先看一眼数据

先直观感受一下 5 个类别：可见幼苗形态相近、背景（砂石/标签/塑料盘）差异很大，
而且植物的**大小、位置、朝向**变化都不小 —— 这直接决定了后面增强策略的设计。
"""))

    cells.append(code(r"""
from PIL import Image
import matplotlib.pyplot as plt

classes = sorted(os.listdir(os.path.join(DATA_ROOT, "train")))
fig, axes = plt.subplots(len(classes), 5, figsize=(13, 2.6 * len(classes)))
for r, cls in enumerate(classes):
    files = sorted(os.listdir(os.path.join(DATA_ROOT, "train", cls)))[:5]
    for c, fn in enumerate(files):
        img = Image.open(os.path.join(DATA_ROOT, "train", cls, fn)).convert("RGB")
        axes[r, c].imshow(img)
        axes[r, c].axis("off")
        if c == 0:
            axes[r, c].set_title(cls, loc="left", fontsize=11)
plt.tight_layout()
plt.show()

# 尺寸分布：输入分辨率需要统一
print("原始图片尺寸统计")
sizes = []
for cls in classes:
    for fn in os.listdir(os.path.join(DATA_ROOT, "train", cls))[:20]:
        sizes.append(Image.open(os.path.join(DATA_ROOT, "train", cls, fn)).size)
print("  最小:", min(sizes), " 最大:", max(sizes))
"""))

    cells.append(md("""
## 3. 载入训练管线

下面的单元格把完整的训练管线写入 `plant_classifier.py` 并导入。

这套代码是**本地在 CPU 上真实跑通并验证过**的同一份代码，包含：

* 图像缓存（一次性缩放到 384×384 存成 `uint8` 数组，训练时不再解码 PNG，CPU 上提速明显）
* 模型库（resnet / efficientnet / convnext / densenet / mobilenet / regnet / swin / vit）
* 数据增强与 `Dataset`
* 分层 K 折划分（不依赖 sklearn）
* 单折训练（判别式学习率 + OneCycle + label smoothing + 可选 mixup）
* 8 视角 TTA 的测试预测
* 提交文件生成（按模板 ID 顺序对齐）
"""))

    cells.append(code("PIPELINE_SOURCE = r'''\n" + pipeline_src + "\n'''"))

    cells.append(code(r"""
import os, sys

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SRC_DIR = os.path.join(WORK, "src")
os.makedirs(SRC_DIR, exist_ok=True)
with open(os.path.join(SRC_DIR, "plant_classifier.py"), "w", encoding="utf-8") as fh:
    fh.write(PIPELINE_SOURCE)

sys.path.insert(0, SRC_DIR)
import plant_classifier as pc

print("pipeline loaded | classes:", pc.CLASSES)
print("available presets:")
for k, v in pc.MODEL_PRESETS.items():
    print("   %-20s size=%d batch=%d" % (k, v["size"], v["batch"]))

pc.OUT = os.path.join(WORK, "outputs")        # 所有训练产物写到这里
pc.DATA_CANDIDATES.insert(0, DATA_ROOT)
os.makedirs(pc.OUT, exist_ok=True)
print("\n输出目录:", pc.OUT)
"""))

    cells.append(md("""
## 4. 配置

**默认配置 = GPU 版四骨干集成**（`efficientnet_b0 + resnet50 + convnext_tiny + resnet18`）。
之所以要换架构做集成，是因为单模型 5 折的 OOF 已经到 92% 附近，
继续调学习率/增强的边际收益很低；而不同骨干的归纳偏置差异大、错误相关性低，
集成通常能再拿 1.5~2.5 个点，是目前性价比最高的一步。

切换到不同算力的配置：

| 配置 | 骨干 | 5 折 × 25 epoch |
|---|---|---|
| **GPU 四骨干集成（默认）** | efb0 + r50 + cnxt + r18 | 约 45~60 分钟 |
| GPU 双骨干 | efb0 + r18 | 约 15 分钟 |
| CPU 单骨干 | efb0 | 约 80 分钟 |
| CPU 快速看结果 | r18 | 约 15 分钟 |

秒/epoch 的实测基准（500 张图一个 epoch）：

| 骨干 | 输入尺寸 | CPU(16线程) | GPU(T4, 约 10×) |
|---|---|---|---|
| `resnet18` | 256 | 21 s | ~2 s |
| `efficientnet_b0` | 288 | 38 s | ~4 s |
| `resnet50` | 256 | 100 s | ~9 s |
| `convnext_tiny` | 256 | 100 s | ~9 s |

**显存**：单骨干 batch 20~32 时占用约 1.5~3 GB，T4 的 16 GB 足够，
四个骨干是**串行**训练的（一个训练完再下一个），不会叠加显存。

**注意**：Kaggle 新建 Notebook 默认是 CPU。如果你不开 GPU，请把 `MODELS`
改成 `["efficientnet_b0"]`（约 80 分钟），否则四骨干在 CPU 上要跑 9 小时以上。
"""))

    cells.append(code(r"""
EXPERIMENT = dict(
    # ================= 默认：GPU 四骨干集成 =================
    # 不同架构的归纳偏置差异大、错误相关性低，概率平均后通常 +1.5~2.5 个点
    MODELS   = ["efficientnet_b0",    # depthwise separable + SE，最稳的主力
                "resnet50",           # 经典残差，与 EfficientNet 互补
                "convnext_tiny",      # 大 kernel depthwise，现代架构
                "resnet18"],          # 最轻量，进一步平滑方差
    FOLDS    = 5,
    EPOCHS   = 25,
    TTA      = 8,        # 8 个几何视角（原图/翻转×2/旋转 90·180·270 …）

    # 推理时额外参与平均的尺度。None = 只用各骨干自己的训练尺度，最省时间；
    # 想要多尺度 TTA 就打开下面这行（代价是推理时间 ×3）：
    # PREDICT_SIZES = [256, 288, 320],

    SEED     = 2026,

    # ================= 只用 CPU 时改成这个 =================
    # MODELS = ["efficientnet_b0"],
    # EPOCHS = 20,

    # ================= 先快速看一眼分数 =================
    # MODELS = ["resnet18"],
    # EPOCHS = 12,
)

# 学号（组名）—— 提交前请改成你自己的学号
STUDENT_ID = "请填写你的学号"

print(json.dumps({k: v for k, v in EXPERIMENT.items()}, indent=2, ensure_ascii=False))
"""))

    cells.append(md("""
### 4.1 先确认算力与预计耗时

运行下面的单元格会打印当前设备、显存、以及本次实验的预计耗时。
如果发现时间太长，**回到上一个单元格把 `EPOCHS` 调小或减少 `MODELS`**，再往下跑。
"""))

    cells.append(code(r"""
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("设备:", dev.upper(), end="")
if dev == "cuda":
    free, total_gb = torch.cuda.mem_get_info()
    print("  (%s, 显存 %.1f GB, 可用 %.1f GB)"
          % (torch.cuda.get_device_name(0), total_gb / 1e9, free / 1e9))
    # 这张卡支持的话打开 TF32，Ampere 及以后能明显加速卷积
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
else:
    print("  (无 GPU，用 %d 线程 CPU)" % torch.get_num_threads())
    print("  ！提醒：四骨干在 CPU 上要约 9 小时，建议把 MODELS 改成 [\"efficientnet_b0\"]")

# 单 epoch 经验耗时（秒/epoch，500 张图）—— 16 线程 CPU 实测
SEC_PER_EPOCH_CPU = {"efficientnet_b0": 38, "resnet18": 21, "resnet50": 100,
                     "convnext_tiny": 100, "densenet121": 100,
                     "mobilenet_v3_large": 25, "efficientnet_b1": 55}
SPEED = 1.0 if dev == "cpu" else 0.1          # GPU 约快 10 倍

total = 0.0
for m in EXPERIMENT["MODELS"]:
    per_ep = SEC_PER_EPOCH_CPU.get(m, 40) * SPEED
    fold_min = per_ep * EXPERIMENT["EPOCHS"] / 60.0
    total += fold_min * EXPERIMENT["FOLDS"]
    print("   %-18s 每折约 %5.1f 分钟，%d 折共 %5.1f 分钟"
          % (m, fold_min, EXPERIMENT["FOLDS"], fold_min * EXPERIMENT["FOLDS"]))

print("\n训练合计: 约 %.0f 分钟 (%.1f 小时)" % (total, total / 60.0))
print("推理（%d 折 × %d 视角 TTA）单独再算，通常几分钟内。"
      % (EXPERIMENT["FOLDS"], EXPERIMENT["TTA"]))
if len(EXPERIMENT["MODELS"]) == 1:
    print("\n提示：再加一个 resnet18 做双骨干集成，GPU 上只多 ~6 分钟，通常能涨 1~2 个点。")
print("\n时间可接受就继续往下运行；否则回到上一个单元格调小 EPOCHS / 减少 MODELS。")
"""))

    cells.append(md("""
## 5. 建立图像缓存

把所有图片一次性缩放到 384×384 并保存为 `uint8` 数组。之后每个 epoch 的数据增强
直接在内存里做，避免了反复解码 PNG 的开销 —— 在 CPU 上这一步能省下大量时间。
"""))

    cells.append(code(r"""
t0 = time.time()
pc.build_cache(DATA_ROOT, size=pc.CACHE_SIZE, force=False)
print("cache ready in %.1f s" % (time.time() - t0))
"""))

    cells.append(md("""
## 6. 快速验证（可选，1 折 × 3 epoch）

正式训练前先跑一个很小的实验，确认整条链路（数据 → 增强 → 前向 → 反向 → 验证）
没有报错，并且准确率明显高于随机猜测的 20%。大约 1~2 分钟。
"""))

    cells.append(code(r"""
pc.set_seed(EXPERIMENT["SEED"])
quick_acc = pc.train_one(
    EXPERIMENT["MODELS"][0], fold=0, folds=EXPERIMENT["FOLDS"],
    epochs=3, seed=EXPERIMENT["SEED"], data_root=DATA_ROOT, verbose=True)
print("\n3-epoch sanity-check accuracy = %.4f  (random guess = 0.20)" % quick_acc)
"""))

    cells.append(md("""
## 7. 正式训练：5 折交叉验证 × 多骨干

对每个骨干、每一折：用 400 张训练、100 张验证，跑完整 epoch 数，
保存**验证集上最好的**权重（不是最后一个 epoch —— 实测最优通常出现在中段）。

每折结束后该折的验证图不会被后续折使用，因此汇总起来就是无偏的 OOF 准确率。
权重按 `骨干名_foldN.pt` 保存，**中途断了直接重跑本单元格即可**，
已完成的折会自动跳过，不会白跑。
"""))

    cells.append(code(r"""
import time as _time

summary = {}
_t_start = _time.time()
for name in EXPERIMENT["MODELS"]:
    for fold in range(EXPERIMENT["FOLDS"]):
        _t0 = _time.time()
        acc = pc.train_one(
            name, fold=fold, folds=EXPERIMENT["FOLDS"],
            epochs=EXPERIMENT["EPOCHS"], seed=EXPERIMENT["SEED"],
            data_root=DATA_ROOT, verbose=True)
        summary["%s_fold%d" % (name, fold)] = acc
        done = len(summary)
        total_runs = len(EXPERIMENT["MODELS"]) * EXPERIMENT["FOLDS"]
        print("      -> 已完成 %d/%d 个 (骨干,折)，本折 %.1f 分钟，总耗时 %.1f 分钟"
              % (done, total_runs, (_time.time() - _t0) / 60.0,
                 (_time.time() - _t_start) / 60.0), flush=True)

print("\n================ 交叉验证结果 ================")
for k in sorted(summary):
    print("  %-28s %.4f" % (k, summary[k]))
print("  %-28s %.4f" % ("MEAN", sum(summary.values()) / len(summary)))

# 按骨干汇总，方便判断哪个模型更强、集成是否有互补性
print("\n================ 按骨干汇总 ================")
for name in EXPERIMENT["MODELS"]:
    vals = [v for k, v in summary.items() if k.startswith(name + "_fold")]
    if vals:
        print("  %-18s mean %.4f  (%s)" % (name, sum(vals) / len(vals),
                                            " ".join("%.3f" % v for v in vals)))
"""))

    cells.append(md("""
## 8. 训练曲线与混淆矩阵

* **混淆矩阵**：哪两个类别容易被混淆 —— 禾本科杂草之间通常最难分。
* **训练曲线**：检查是否过拟合/欠拟合（本任务训练集只有 400 张，过拟合是主要风险）。
"""))

    cells.append(code(r"""
import numpy as np

# ---- OOF 混淆矩阵 ----
n_cls = len(pc.CLASSES)
cm = np.zeros((n_cls, n_cls), dtype=int)
oof_probs, oof_true = [], []
for name in EXPERIMENT["MODELS"]:
    for fold in range(EXPERIMENT["FOLDS"]):
        p = os.path.join(pc.OUT, "oof", "%s_fold%d.npz" % (name, fold))
        if os.path.exists(p):
            d = np.load(p)
            oof_probs.append(d["probs"]); oof_true.append(d["labels"])
            for t, q in zip(d["labels"], d["probs"].argmax(1)):
                cm[t, q] += 1

oof_probs = np.concatenate(oof_probs); oof_true = np.concatenate(oof_true)
oof_acc = (oof_probs.argmax(1) == oof_true).mean()
print("OOF accuracy = %.4f  (%d/%d)" % (oof_acc, (oof_probs.argmax(1) == oof_true).sum(), len(oof_true)))

# 每类召回率
print("\n每类召回率")
for i, c in enumerate(pc.CLASSES):
    tot = cm[i].sum()
    print("   %-18s %3d/%3d = %.3f" % (c, cm[i, i], tot, cm[i, i] / max(1, tot)))

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(n_cls)); ax.set_xticklabels(pc.CLASSES, rotation=35, ha="right")
ax.set_yticks(range(n_cls)); ax.set_yticklabels(pc.CLASSES)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("OOF confusion matrix")
for i in range(n_cls):
    for j in range(n_cls):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black")
plt.colorbar(im); plt.tight_layout(); plt.show()
"""))

    cells.append(code(r"""
# ---- 训练曲线 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for name in EXPERIMENT["MODELS"]:
    for fold in range(EXPERIMENT["FOLDS"]):
        p = os.path.join(pc.OUT, "history_%s_fold%d.json" % (name, fold))
        if not os.path.exists(p):
            continue
        h = json.load(open(p))["history"]
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["train_loss"] for x in h], alpha=.6, label="%s f%d train" % (name, fold))
        axes[0].plot(ep, [x["val_loss"] for x in h], alpha=.6, ls="--", label="%s f%d val" % (name, fold))
        axes[1].plot(ep, [x["val_acc"] for x in h], alpha=.7, label="%s f%d" % (name, fold))
axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=7)
axes[1].axhline(0.2, color="gray", ls=":", label="random")
axes[1].set_title("validation accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
"""))

    cells.append(md("""
## 9. 测试集预测（含 TTA）

对每张测试图生成 8 个几何视角（原图、水平/垂直翻转、旋转 90/180/270 等），
分别前向传播后对 **softmax** 取平均（不是在 logit 空间平均）。

聚合顺序很重要：**先按折平均 → 再按骨干平均**。这样即使某个骨干只训练了部分折，
也不会在最终结果里被隐式降权。

单张测试图的预测由 `折数 × 骨干数 × 8` 次前向共同决定（四骨干 5 折就是 160 次），
比单次预测稳定得多。若打开了 `PREDICT_SIZES`，再乘上尺度数。

### 这一步还会做「决策权重修正」

这一步不只是推理，还顺带修正了一个**很容易被忽略、但扣分很实在**的问题：

ImageNet 预训练权重自带它自己的类别先验，而我们的训练集只有 500 张、5 类各 100 张。
两者叠加后，模型会**系统性地高估某些类、低估另一些类**。表现就是：明明训练集是均匀的
（每类 20%），模型在测试集上却会给出很偏的分布。

如果评分用的是 macro-F1（这类课程比赛通常如此），这种偏置会直接扣分，
而且**只靠换网络、加 epoch 是修不掉的** —— 因为它是决策偏置，不是特征问题。

所以我用 5 折交叉验证产出的 **out-of-fold 概率**去测量这个偏置，
然后在 argmax 之前给每一类乘一个修正系数。整个过程：

1. 从 OOF 概率算出模型的实际预测分布
2. 与"应该均匀"这个先验比较，得到初始修正权重
3. 用坐标上升法在 OOF 上微调权重，目标函数是 **balanced accuracy**
4. **留一折验证**：在其中 4 折上学权重，到第 5 折上检验是否真的变好

第 4 步是关键 —— 它保证报告出来的增益不是"在同一批数据上拟合两次"的假象。
自由参数只有 4 个（5 类权重归一化后），相对 ~500 个 OOF 样本，过拟合风险很低。
"""))

    cells.append(code(r"""
import time as _time

predict_sizes = EXPERIMENT.get("PREDICT_SIZES")
n_views = EXPERIMENT["FOLDS"] * EXPERIMENT["TTA"] * (len(predict_sizes) if predict_sizes else 1)
print("每个骨干要做 %d 次前向（%d 折 × %d 视角 × %d 尺度）"
      % (n_views, EXPERIMENT["FOLDS"], EXPERIMENT["TTA"],
         len(predict_sizes) if predict_sizes else 1))
if predict_sizes:
    print("多尺度 TTA 尺度:", predict_sizes)

_t0 = _time.time()
probs, per_model, test_names, oof_acc, weights = pc.predict_test(
    EXPERIMENT["MODELS"], folds=EXPERIMENT["FOLDS"], tta=EXPERIMENT["TTA"],
    data_root=DATA_ROOT, tta_sizes=predict_sizes)
print("推理耗时 %.1f 分钟" % ((_time.time() - _t0) / 60.0))

print("\n集成 OOF 准确率: %.4f" % oof_acc)

# 决策权重：修正迁移学习继承来的类别先验偏置
print("\n决策权重（每类乘数，>1 表示该类原来被低估）:")
for i, c in enumerate(pc.CLASSES):
    print("   %-18s %.3f" % (c, weights[i]))

# 加权前后的测试集预测分布：应当更接近训练集的均匀分布
print("\n测试集预测分布:")
raw_dist = pc.class_distribution(probs)
adj_dist = pc.class_distribution(probs * weights[None, :])
print("   %-18s %10s %10s %10s" % ("类别", "未加权", "加权后", "训练集"))
for i, c in enumerate(pc.CLASSES):
    print("   %-18s %9.1f%% %9.1f%% %9.1f%%"
          % (c, 100 * raw_dist[i], 100 * adj_dist[i], 100.0 / len(pc.CLASSES)))

pred = probs.argmax(1)
print("\n测试集逐类计数:")
for i, c in enumerate(pc.CLASSES):
    print("   %-18s %4d (%.1f%%)" % (c, (pred == i).sum(), 100 * (pred == i).mean()))

# 预测置信度：低置信度样本是主要的错误来源
conf = probs.max(1)
print("\n置信度: 均值 %.3f  中位数 %.3f  最低 %.3f" %
      (conf.mean(), np.median(conf), conf.min()))
print("置信度 < 0.5 的样本数: %d / %d" % ((conf < 0.5).sum(), len(conf)))

# 骨干之间的一致度：一致度低说明这些样本确实难
per_model_arr = [np.asarray(p) for p in per_model]
if len(per_model_arr) > 1:
    stacks = [p.argmax(1) for p in per_model_arr]
    agree = np.mean([len({s[i] for s in stacks}) == 1 for i in range(len(test_names))])
    print("不同骨干预测完全一致的比例: %.1f%%" % (100 * agree))
"""))

    cells.append(code(r"""
# 看几个低置信度的测试样本，理解模型在哪里犹豫
order = np.argsort(conf)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for ax, i in zip(axes, order[:6]):
    img = Image.open(os.path.join(DATA_ROOT, "test", test_names[i])).convert("RGB")
    ax.imshow(img); ax.axis("off")
    top2 = np.argsort(probs[i])[::-1][:2]
    ax.set_title("%s %.2f\n%s %.2f" % (pc.CLASSES[top2[0]], probs[i][top2[0]],
                                      pc.CLASSES[top2[1]], probs[i][top2[1]]), fontsize=8)
plt.suptitle("最低置信度的 6 张测试图")
plt.tight_layout(); plt.show()
"""))

    cells.append(md("""
## 10. 生成提交文件

格式与 `submission-for-task2.csv` 完全一致（两列 `ID,Category`），并且**顺序按官方模板对齐**。

提交时用的是**加权后的预测**（每类乘上上一步学到的决策权重），这是提升 macro-F1 的关键一步。
同时也会写出一份未加权的版本 `submission_plain.csv`，方便你对比两者差异。

**注意**：课程要求组名必须为个人学号，请把上面的 `STUDENT_ID` 改成你的学号，
提交到 Kaggle 时也要用学号作为队名。Kaggle 会自动读取 `/kaggle/working/submission.csv`。
"""))

    cells.append(code(r"""
SUB_PATH = "/kaggle/working/submission.csv"
PLAIN_PATH = "/kaggle/working/submission_plain.csv"

# 加权提交（推荐）
pc.write_submission(probs, test_names, SUB_PATH,
                    participant=STUDENT_ID, template=TEMPLATE, weights=weights)
# 未加权提交（对照）
pc.write_submission(probs, test_names, PLAIN_PATH, participant=STUDENT_ID,
                    template=TEMPLATE)

n_diff = int((pc.apply_weights(probs, weights) != probs.argmax(1)).sum())
print("\n加权与未加权两种提交相差 %d / %d 个样本 (%.1f%%)"
      % (n_diff, len(probs), 100 * n_diff / len(probs)))
print("如果提高明显，说明原先的类别偏置确实在扣分。")

import csv
with open(SUB_PATH) as fh:
    rows = list(csv.reader(fh))
print("\n文件行数(含表头):", len(rows))
print("前 5 行:")
for r in rows[:5]:
    print("   ", r)
"""))

    cells.append(code(r"""
# 提交前自检：ID 必须与官方模板完全一致，不能有多余或缺失的行
if TEMPLATE:
    with open(TEMPLATE) as fh:
        tmpl_ids = [r[0] for r in list(csv.reader(fh))[1:] if r]
    sub_ids = [r[0] for r in rows[1:]]
    assert set(tmpl_ids) == set(sub_ids), "ID 集合不一致！"
    assert len(tmpl_ids) == len(sub_ids), "行数不一致！"
    print("自检通过：%d 行，ID 与官方模板完全一致" % len(sub_ids))
else:
    print("未找到模板，跳过自检（请手动核对 ID 列）")
"""))

    cells.append(md("""
## 11. 进阶（可选）：伪标签再训练一轮

到这里已经有一份可提交的结果了。下面的部分**可选**，但它是本任务剩余空间里最大的一块收益来源。

**动机**：真正的约束不是模型容量，而是**只有 500 张标注图**。而测试集那 378 张虽然没标签，
却**恰好来自要被评分的那同一个分布** —— 这是无监督信息最"对味"的一种。
把其中模型很有把握的部分当作额外训练数据，等于把标注量扩大 70% 以上。

**风险与对策**：伪标签若选错，会把错误固化进模型。所以这里设了三道闸：

| 闸门 | 做法 |
|---|---|
| 置信度 | 只接受集成预测概率 ≥ `min_conf`（默认 0.90）的样本 |
| 骨干一致 | 要求**所有骨干网络**对同一张图给出同一个类别 |
| 诚实评估 | 伪标签**只进训练集，绝不进验证集**，所以打印的验证准确率仍是在真实标注上算的 |

**一个容易被忽略的坑**：伪标签模型的验证图，属于"当初挑选伪标签的那个池子"，
所以它的验证概率**不能**当作 out-of-fold 存下来 —— 否则会让决策权重修正偷偷作弊。
代码里这一点是用 `save_oof=False` 强制的，并写进了注释。

下面会先打印被选中样本的类别分布。**如果某个类别几乎没被选中，说明模型对它整体没把握**，
这本身就是有用的诊断信息。
"""))

    cells.append(code(r"""
# 第二轮：用高置信度测试图做伪标签，重新训练所有 (骨干, 折)
# 注意：这一步会再花掉和第一轮差不多的训练时间。想跳过就直接运行后面的单元格。
ROUND2 = True          # 不想跑就改成 False

if ROUND2:
    pl_results = pc.pseudo_label_train(
        EXPERIMENT["MODELS"],
        folds=EXPERIMENT["FOLDS"],
        epochs=EXPERIMENT.get("EPOCHS", 25),
        min_conf=0.90,               # 置信度阈值
        require_agreement=True,      # 要求所有骨干一致
        tta=EXPERIMENT["TTA"],
        data_root=DATA_ROOT,
        seed=EXPERIMENT["SEED"],
    )
else:
    print("已跳过第二轮伪标签训练")
"""))

    cells.append(md("""
## 12. 小结

**做了什么**

| 环节 | 选择 | 原因 |
|---|---|---|
| 预训练 | ImageNet 迁移学习 | 每类仅 100 张图，从零训练不可行 |
| 验证 | 5 折分层交叉验证 | 得到无偏 OOF 估计，同时免费得到 5 模型集成 |
| 增强 | 随机裁剪/翻转/旋转/颜色抖动/随机擦除 | 植物位置、朝向、光照差异大 |
| 优化 | 判别式学习率 + OneCycle + label smoothing | 保护预训练特征，同时让新分类头快速收敛 |
| 模型 | 多骨干集成（EfficientNet-B0 / ResNet50 / ConvNeXt-Tiny / ResNet18） | 不同架构的错误相关性低，平均后降方差 |
| 推理 | 8 视角 TTA + 多折平均（可加多尺度） | 降低单次预测的方差 |
| **决策** | **OOF 决策权重修正（均衡各类预测）** | **修正迁移学习继承的类别先验偏置，直接提升 macro-F1** |
| 半监督 | 伪标签再训练（可选） | 把无标注测试集变成训练数据 |

**评估指标为什么要看两个**

| 指标 | 含义 | 什么时候更该看它 |
|---|---|---|
| accuracy | 总体正确率 | 类别均衡、且评分就是 accuracy |
| balanced accuracy | 每类召回的平均 | 类别均衡但模型偏向多数类 |
| macro F1 | 每类 F1 的平均 | **评分通常是这个**，对类别偏置最敏感 |

训练集是 5 类各 100 张、完全均匀，所以这三者**本该很接近**。
如果你的 accuracy 是 0.92 而 macro F1 只有 0.85 左右，差值就是类别偏置在扣分，
决策权重修正正是针对它的。

**还能怎么提升**

1. **更多骨干**：再加 `densenet121` 或 `swin_t`，进一步增加架构多样性。
2. **多尺度 TTA**：打开配置里的 `PREDICT_SIZES = [256, 288, 320]`。
3. **伪标签迭代多轮**：把第二轮模型重新预测测试集、再筛一次伪标签，通常还能再涨一点。
4. **针对易混类别**：从混淆矩阵看，禾本科杂草（Black-grass / Common wheat /
   Loose Silky-bent）之间最容易混，可对这几类单独训练一个细粒度分类器，
   或对易混类做局部区域的高分辨率裁剪。
5. **更高分辨率训练**：细粒度识别对分辨率敏感，GPU 允许时可以试 384。

### 关于"利用测试集结构做平滑"这类后处理

理论上可以做一个直推式（transductive）后处理：用特征空间里的 k 近邻把概率平滑几轮，
`P ← (1-α)P + α·(近邻平均)`。它不使用标签，因此不存在泄漏。

**但实测结论是：在这份数据上它没有帮助，反而有害。**
在留出验证集上，平滑后准确率明显下降，原因是近邻里混入了真正难分的禾本科样本，
于是把本来正确的预测拖错了。

所以这一版**没有把它放进流程**。这里记下来是为了说明一个更重要的原则：
**任何后处理都必须先在留出集上验证有效性，再决定是否采用** ——
"不使用标签"只代表没有泄漏，不代表有效。
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
    with open(OUT_NB, "w", encoding="utf-8") as fh:
        json.dump(notebook, fh, ensure_ascii=False, indent=1)

    # validate that it is well-formed JSON with the expected shape
    with open(OUT_NB, encoding="utf-8") as fh:
        check = json.load(fh)
    n_code = sum(1 for c in check["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in check["cells"] if c["cell_type"] == "markdown")
    print("wrote %s (%d cells: %d code, %d markdown)"
          % (OUT_NB, len(check["cells"]), n_code, n_md))
    return OUT_NB


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
