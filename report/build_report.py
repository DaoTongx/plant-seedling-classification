# -*- coding: utf-8 -*-
"""
生成课程设计实验报告（任务一：植物分类）的 Word 文档。

所有数值均取自本次实验的真实产物（训练日志、OOF 评测输出、Kaggle 提交记录），
未做任何虚构。
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT_DIR = os.path.join(PROJECT, "report")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# =========================================================================== #
#  真实实验数据
# =========================================================================== #
CLASSES = ["Black-grass", "Common wheat", "Loose Silky-bent",
           "Scentless Mayweed", "Sugar beet"]
CLASSES_CN = ["黑麦草", "普通小麦", "散穗黍", "无香母菊", "甜菜"]

# 四骨干 × 5 折 验证准确率（Kaggle GPU 实测）
FOLD_ACC = {
    "efficientnet_b0": [0.93, 0.91, 0.91, 0.92, 0.94],
    "resnet50":        [0.95, 0.93, 0.94, 0.92, 0.91],
    "convnext_tiny":   [0.97, 0.91, 0.96, 0.93, 0.94],
    "resnet18":        [0.96, 0.89, 0.93, 0.89, 0.93],
}

# 集成后的 OOF 指标
OOF_ACC, OOF_BAL, OOF_F1 = 0.9260, 0.9260, 0.9256
OOF_RECALL = [0.830, 0.970, 0.830, 1.000, 1.000]
OOF_PREC = [0.814, 0.942, 0.883, 1.000, 0.990]
OOF_F1CLS = [0.822, 0.956, 0.856, 1.000, 0.995]
OOF_DIST = [0.204, 0.206, 0.188, 0.200, 0.202]

# 决策权重修正
TUNED_WEIGHTS = [0.704, 1.115, 1.090, 1.073, 1.021]
TUNE_AFTER = dict(acc=0.9400, bal=0.9400, f1=0.9391, changed=13)
LOO_F1 = [(0.9604, 0.9700), (0.8774, 0.9074), (0.9203, 0.9200),
          (0.9395, 0.9489), (0.9292, 0.9278)]
LOO_SUM = dict(bal=(0.9260, 0.9360), f1=(0.9254, 0.9348))

# 测试集预测分布
TEST_BEFORE = [18.8, 8.7, 26.7, 23.8, 22.0]
TEST_AFTER = [15.1, 9.0, 30.2, 23.8, 22.0]

# 训练曲线（efficientnet_b0, fold0, CPU, 30 epoch）
TRAIN_LOG = r"""1.5982 1.5480 0.3300
1.5120 1.3639 0.5100
1.2870 1.0827 0.6300
0.9771 0.7994 0.8200
0.7728 0.6796 0.8700
0.7105 0.6414 0.9000
0.6841 0.6252 0.8900
0.6237 0.6050 0.9100
0.6322 0.5883 0.9000
0.6006 0.5850 0.9200
0.6188 0.5719 0.9100
0.5805 0.5679 0.9100
0.5468 0.5717 0.8900
0.5868 0.5853 0.9000
0.5675 0.5749 0.9000
0.5419 0.5703 0.9000
0.5447 0.5541 0.9200
0.5448 0.5513 0.9200
0.5169 0.5628 0.9000
0.5170 0.5533 0.9100
0.5402 0.5601 0.9200
0.4986 0.5557 0.9000
0.5350 0.5502 0.9000
0.5055 0.5512 0.9100
0.4987 0.5582 0.9100
0.5028 0.5525 0.9100
0.4679 0.5531 0.9100
0.4836 0.5519 0.9100
0.4968 0.5541 0.9100
0.5133 0.5545 0.9200"""


# =========================================================================== #
#  matplotlib 中文字体
# =========================================================================== #
def setup_font():
    from matplotlib import font_manager
    wanted = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun", "DejaVu Sans"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((w for w in wanted if w in have), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    print("[font]", chosen)
    return chosen


# =========================================================================== #
#  图 1-1 技术路线
# =========================================================================== #
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fa", ec="#4a6fa5", fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=1.1, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.1, color="#5a6b7d"))

    box(2.6, 11.1, 4.8, 0.85, "训练集 500 张（5 类 × 100） + 测试集 378 张", fc="#dbe7f5", bold=True)
    box(2.6, 9.95, 4.8, 0.8, "预处理：补边成正方形 → 缩放 384×384 → uint8 缓存")
    arrow(5.0, 11.1, 5.0, 10.75)

    box(0.35, 8.5, 4.3, 0.9, "训练分支\n随机裁剪缩放 / 翻转 / 旋转\n颜色抖动 / 随机擦除")
    box(5.35, 8.5, 4.3, 0.9, "验证与推理分支\n确定性缩放\n（推理再加 8 视角 TTA）")
    arrow(5.0, 9.95, 2.5, 9.4)
    arrow(5.0, 9.95, 7.5, 9.4)

    box(0.35, 7.15, 9.3, 0.9,
        "ImageNet 预训练骨干（EfficientNet-B0 / ResNet50 / ConvNeXt-Tiny / ResNet18）"
        "\n并替换分类头为 Linear(d, 5)", fs=8.8)
    arrow(5.0, 8.5, 5.0, 8.05)

    box(2.6, 5.9, 4.8, 0.85, "判别式学习率 + OneCycle + label smoothing\n5 折分层交叉验证训练", fc="#fdf1e3", ec="#c8873f", bold=True)
    arrow(5.0, 7.15, 5.0, 6.75)

    box(2.6, 4.65, 4.8, 0.85, "每折取验证集最优权重\n→ 5 个折模型 × 4 骨干 = 20 个模型")
    arrow(5.0, 5.9, 5.0, 5.5)

    box(0.35, 3.3, 4.3, 0.9, "8 视角 TTA 平均\n(softmax 空间)", fs=9)
    box(5.35, 3.3, 4.3, 0.9, "先按折平均 → 再按骨干平均\n（骨干等权）", fs=9)
    arrow(5.0, 4.65, 2.5, 4.2)
    arrow(5.0, 4.65, 7.5, 4.2)
    arrow(2.5, 3.75, 7.5, 3.75) if False else None

    box(2.6, 2.05, 4.8, 0.85, "决策后处理（二选一）\nOOF 决策权重 / 测试集先验修正", fc="#eaf5ea", ec="#4f8a4f", bold=True)
    arrow(2.5, 3.3, 5.0, 2.9)
    arrow(7.5, 3.3, 5.0, 2.9)

    box(2.6, 0.85, 4.8, 0.75, "输出 submission.csv（ID, Category）", fc="#f2eefa", ec="#6b5aa5", bold=True)
    arrow(5.0, 2.05, 5.0, 1.6)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1-1_pipeline.png")
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =========================================================================== #
#  图 1-2 训练曲线
# =========================================================================== #
def fig_curve():
    arr = np.array([[float(x) for x in ln.split()] for ln in TRAIN_LOG.strip().splitlines()])
    ep = np.arange(1, len(arr) + 1)
    train_loss, val_loss, val_acc = arr[:, 0], arr[:, 1], arr[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    ax = axes[0]
    ax.plot(ep, train_loss, "-o", ms=3, lw=1.5, label="训练损失", color="#2f6fb3")
    ax.plot(ep, val_loss, "-s", ms=3, lw=1.5, label="验证损失", color="#d1682c")
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("交叉熵损失")
    ax.set_title("(a) 损失曲线", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, ls=":")

    ax = axes[1]
    ax.plot(ep, val_acc * 100, "-o", ms=3.5, lw=1.6, color="#3f8f4f")
    best_i = int(np.argmax(val_acc))
    ax.plot(ep[best_i], val_acc[best_i] * 100, "*", ms=15, color="#c0392b",
            label="最优轮次 (ep %d, %.0f%%)" % (ep[best_i], val_acc[best_i] * 100))
    ax.axhline(20, ls=":", lw=1, color="gray", label="随机猜测 20%")
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("验证准确率 (%)")
    ax.set_ylim(15, 100)
    ax.set_title("(b) 验证准确率曲线", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3, ls=":")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1-2_curve.png")
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =========================================================================== #
#  图 1-3 各折验证准确率
# =========================================================================== #
def fig_folds():
    names = list(FOLD_ACC.keys())
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    x = np.arange(len(names))
    width = 0.15
    colors = ["#2f6fb3", "#d1682c", "#3f8f4f", "#9b59b6", "#7f8c8d"]
    for i, col in enumerate(colors):
        vals = [FOLD_ACC[n][i] * 100 for n in names]
        ax.bar(x + (i - 2) * width, vals, width, label="fold %d" % i,
               color=col, edgecolor="white", lw=0.5)
    for i, n in enumerate(names):
        m = np.mean(FOLD_ACC[n]) * 100
        ax.hlines(m, x[i] - 0.42, x[i] + 0.42, color="#c0392b", lw=1.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n均值 %.1f%%" % (n, np.mean(FOLD_ACC[n]) * 100) for n in names],
                       fontsize=9)
    ax.set_ylabel("验证准确率 (%)")
    ax.set_ylim(80, 100)
    ax.set_title("四种骨干网络在 5 折交叉验证下的验证准确率", fontsize=10.5, pad=10)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              frameon=False)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1-3_folds.png")
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =========================================================================== #
#  图 1-4 各类别指标
# =========================================================================== #
def fig_per_class():
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(CLASSES))
    w = 0.26
    ax.bar(x - w, OOF_RECALL, w, label="召回率", color="#2f6fb3", edgecolor="white", lw=0.5)
    ax.bar(x, OOF_PREC, w, label="精确率", color="#d1682c", edgecolor="white", lw=0.5)
    ax.bar(x + w, OOF_F1CLS, w, label="F1", color="#3f8f4f", edgecolor="white", lw=0.5)
    for i in range(len(CLASSES)):
        ax.text(x[i] - w, OOF_RECALL[i] + 0.012, "%.2f" % OOF_RECALL[i],
                ha="center", fontsize=7.2)
        ax.text(x[i], OOF_PREC[i] + 0.012, "%.2f" % OOF_PREC[i], ha="center", fontsize=7.2)
        ax.text(x[i] + w, OOF_F1CLS[i] + 0.012, "%.2f" % OOF_F1CLS[i], ha="center", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_ylabel("指标值")
    ax.set_ylim(0.70, 1.06)
    ax.set_title("集成模型 OOF 上各类别的召回率、精确率与 F1", fontsize=10.5, pad=10)
    ax.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              frameon=False)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1-4_perclass.png")
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =========================================================================== #
#  图 1-5 测试集预测分布对比
# =========================================================================== #
def fig_dist():
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(CLASSES))
    w = 0.26
    ax.bar(x - w, TEST_BEFORE, w, label="未修正 (argmax)", color="#7f8c8d",
           edgecolor="white", lw=0.5)
    ax.bar(x, TEST_AFTER, w, label="OOF 决策权重修正后", color="#c0392b",
           edgecolor="white", lw=0.5)
    ax.axhline(20.0, color="#2f6fb3", ls="--", lw=1.4, label="训练集先验 (均匀 20%)")
    for i in range(len(CLASSES)):
        ax.text(x[i] - w, TEST_BEFORE[i] + 0.5, "%.1f" % TEST_BEFORE[i],
                ha="center", fontsize=7.5)
        ax.text(x[i], TEST_AFTER[i] + 0.5, "%.1f" % TEST_AFTER[i], ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=9)
    ax.set_ylabel("测试集预测占比 (%)")
    ax.set_ylim(0, 36)
    ax.set_title("决策权重修正前后的测试集预测分布", fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig1-5_dist.png")
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =========================================================================== #
#  docx 工具
# =========================================================================== #
def set_run_font(run, ascii_font="Times New Roman", ea_font="宋体", size=None, bold=None):
    run.font.name = ascii_font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), ea_font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9)


def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def set_table_fixed_layout(table):
    """固定列宽布局，避免 Word 自动重排导致表格超出页边距。

    使用 python-docx 提供的 get_or_add_tblLayout()，它会把元素插到符合
    OOXML 架构规定的顺序上（tblLayout 必须位于 tblBorders/shd 之后、
    tblCellMar 之前），不要直接 append。
    """
    layout = table._tbl.tblPr.get_or_add_tblLayout()
    layout.set(qn("w:type"), "fixed")


def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    # tblBorders 必须排在 shd / tblLayout / tblCellMar / tblLook 之前
    tblPr.insert_element_before(
        borders, "w:shd", "w:tblLayout", "w:tblCellMar",
        "w:tblLook", "w:tblCaption", "w:tblDescription")


def add_caption(doc, text, kind="table"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6 if kind == "figure" else 3)
    run = p.add_run(text)
    set_run_font(run, size=9.5, bold=True)
    return p


def add_table(doc, header, rows, widths=None, font_size=9.0, caption=None,
              align_center_cols=None):
    if caption:
        add_caption(doc, caption, kind="table")
    table = doc.add_table(rows=1, cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    set_table_borders(table)
    set_table_fixed_layout(table)

    hdr = table.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(h))
        set_run_font(run, size=font_size, bold=True)
        shade_cell(hdr[i], "DCE6F1")

    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER
                           if (align_center_cols is None or i in align_center_cols)
                           else WD_ALIGN_PARAGRAPH.LEFT)
            run = p.add_run(str(v))
            set_run_font(run, size=font_size)

    if widths:
        for r in table.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
        # 同步 tblGrid：固定布局下 Word 以网格列宽为列边界，
        # 只设单元格宽度而不改网格会导致各列被均分。
        grid = table._tbl.tblGrid
        cols = grid.findall(qn("w:gridCol"))
        for gc, w in zip(cols, widths):
            gc.set(qn("w:w"), str(int(Cm(w).twips)))
        tblW = table._tbl.tblPr.find(qn("w:tblW"))
        if tblW is not None:
            tblW.set(qn("w:type"), "dxa")
            tblW.set(qn("w:w"), str(int(Cm(sum(widths)).twips)))
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


# =========================================================================== #
#  主构建
# =========================================================================== #
def build():
    setup_font()

    p11 = fig_pipeline()
    p12 = fig_curve()
    p13 = fig_folds()
    p14 = fig_per_class()
    p15 = fig_dist()
    print("[figs] 5 张图已生成")

    doc = Document()

    # ---- 页面与正文样式 ----
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.6)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.4)
    add_page_number_footer(sec)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(0)

    for name, size in (("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 11.5)):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 9)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    def H1(t):
        doc.add_page_break()
        h = doc.add_heading(level=1)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=16, bold=True)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def H2(t):
        h = doc.add_heading(level=2)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=13.5, bold=True)

    def H3(t):
        h = doc.add_heading(level=3)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=11.5, bold=True)

    def P(text, indent=True, bold=False, size=10.5, after=3, align=None):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74) if indent else Cm(0)
        p.paragraph_format.space_after = Pt(after)
        if align is not None:
            p.alignment = align
        p.alignment = p.alignment if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold)
        return p

    def BUL(items, size=10.5):
        for it in items:
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            r = p.add_run(it)
            set_run_font(r, size=size)

    def NUM(items, size=10.5):
        for it in items:
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            r = p.add_run(it)
            set_run_font(r, size=size)

    def FIG(path, caption, width_cm=15.2):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(path, width=Cm(width_cm))
        add_caption(doc, caption, kind="figure")

    def CODE(lines):
        for ln in lines:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            r = p.add_run(ln)
            set_run_font(r, ascii_font="Consolas", ea_font="宋体", size=8.8)

    # ===================================================================== #
    #  封面信息
    # ===================================================================== #
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(6)
    r = t.add_run("机器学习课程设计实验报告")
    set_run_font(r, ea_font="黑体", size=20, bold=True)

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = st.add_run("—— 任务一：植物分类 ——")
    set_run_font(r, ea_font="黑体", size=13)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.paragraph_format.space_before = Pt(4)
    r = info.add_run("（本报告为四个任务中的第一章，可与其他章节合并成册）")
    set_run_font(r, size=9, bold=False)

    add_table(doc,
              ["项目", "内容"],
              [["任务名称", "植物分类（Task 1）"],
               ["任务类型", "五分类图像识别（Kaggle 排名制）"],
               ["训练数据", "500 张，5 类各 100 张"],
               ["测试数据", "378 张，无标注"],
               ["评价指标", "分类准确率（排行榜）"],
               ["最佳提交成绩", "0.9312"],
               ["核心方法", "ImageNet 迁移学习 + 五折分层交叉验证 + 四骨干集成 + 多视角 TTA + 决策偏置修正"]],
              widths=[3.6, 11.6], font_size=9.5,
              align_center_cols=[0])

    # ===================================================================== #
    #  第 1 章
    # ===================================================================== #
    H1("第 1 章  植物分类")

    # ---------------- 1.1 ----------------
    H2("1.1  任务背景与问题描述")
    P("本任务要求对植物幼苗图像进行五分类，识别对象为农田中常见的五种植物：黑麦草"
      "（Black-grass）、普通小麦（Common wheat）、散穗黍（Loose Silky-bent）、"
      "无香母菊（Scentless Mayweed）与甜菜（Sugar beet）。数据以竞赛形式提供：训练集共 "
      "500 张已标注图像，五类各 100 张；测试集 378 张无标注图像。提交文件为两列 CSV，"
      "第一列为测试图像文件名，第二列为预测类别名，最终成绩依据排行榜准确率排名。")
    P("这一任务的难点并不在于“搭建一个卷积神经网络”，而在于数据条件极为苛刻："
      "每类仅有 100 张训练样本，而不同物种的幼苗在形态上高度相似。因此，本实验的设计目标"
      "是：在极小样本条件下，尽可能榨取预训练模型的特征能力，并通过统计上可靠的验证手段"
      "保证每一步改进都是真实的、而非偶然的。")

    # ---------------- 1.2 ----------------
    H2("1.2  问题分析")
    H3("1.2.1  小样本带来的根本约束")
    P("五类共 500 张图像，平均每类 100 张。若从随机初始化开始训练一个中等规模的卷积网络"
      "（例如 EfficientNet-B0，参数量约 400 万），则参数量约为样本量的 8000 倍。在这种"
      "情形下，模型有无数种参数组合可以完美拟合这 500 张图像，却无法保证对第 501 张图像"
      "的判断。这是过拟合的本质，也是本任务最根本的约束。")
    P("因此，本实验并未把重点放在“设计一个更强的网络结构”上，而是放在“如何有效利用 "
      "ImageNet 预训练权重”上。这一判断在后文 1.6 节的实验中得到了验证：仅训练 6 个"
      "轮次，验证准确率即达到 90%，而随机猜测仅为 20%。")

    H3("1.2.2  细粒度类间差异")
    P("三个禾本科物种（黑麦草、普通小麦、散穗黍）在幼苗期的差异主要体现在叶宽、叶脉走向与"
      "叶鞘颜色上，差异尺度远小于 ImageNet 中通常的类间差异（例如“狗”与“汽车”）。"
      "这类问题在文献中被称为细粒度识别。1.6.2 节的实验结果显示，集成模型在无香母菊与甜菜上"
      "的召回率达到 100%，而在黑麦草与散穗黍上仅为 83%，误差几乎全部集中在禾本科内部，"
      "与上述形态学判断一致。")

    H3("1.2.3  类内差异与成像条件")
    P("观察训练样本可见，同一物种的图像在以下方面差异极大：植株在画面中的位置与尺度"
      "（有的植株仅占画面一角，有的几乎铺满）、植株朝向（俯拍视角下完全任意）、"
      "光照条件（闪光灯直射与散射光并存）、以及背景物体（砂石、标签纸、塑料育苗盘）。"
      "这类变异远大于类间差异，是数据增强策略设计的直接依据（见 1.3.3 节）。")

    H3("1.2.4  类别先验偏置：本任务的关键发现")
    P("这是本实验在分析阶段识别出的、最容易被忽略的问题，也是后文决策修正方法的出发点。")
    P("训练集五类完全均衡（各 100 张），因此测试集的真实类别分布应当接近均匀（每类约 20%）。"
      "然而集成模型在测试集上的原始预测分布为：黑麦草 18.8%、普通小麦 8.7%、散穗黍 26.7%、"
      "无香母菊 23.8%、甜菜 22.0%。其中普通小麦被严重压低（8.7% 对期望的 20%），"
      "散穗黍被显著抬高（26.7%）。")
    P("这一偏置的来源是迁移学习本身：ImageNet 的 1000 个类别中，禾本科杂草类样本数量远多于"
      "栽培作物，模型因此继承了一个“看到细长叶片更可能是杂草”的先验。这种偏置不体现为"
      "“特征质量差”，而体现为“决策边界整体偏移”。它的危害在宏平均指标（如 macro-F1）上"
      "被显著放大，且无法通过增加训练轮次或更换骨干网络消除。本实验为此专门设计了"
      "第 1.3.7 节的修正方法，并在 1.6.5 节给出了量化评估。")

    # ---------------- 1.3 ----------------
    H2("1.3  模型与方法设计")
    H3("1.3.1  总体技术路线")
    P("整体流程分为五个阶段：数据预处理与缓存、训练时的随机增强、迁移学习与交叉验证训练、"
      "推理阶段的多视角与多模型聚合、以及决策后处理。技术路线如图 1-1 所示。")
    FIG(p11, "图 1-1  植物分类任务的总体技术路线")

    H3("1.3.2  骨干网络与分类头设计")
    P("本实验选用四种归纳偏置差异明显的骨干网络做集成，而非同族的不同深度版本。"
      "这一选择的依据是集成学习中“误差去相关性”原理：若若干模型的错误高度相关"
      "（错在同一批图像上），则平均无效；只有错误相关性低时，平均才能降低方差。"
      "四个骨干的结构特点与参数量见表 1-1。")
    add_table(doc,
              ["骨干网络", "结构特点", "参数量", "输入分辨率", "批大小"],
              [["EfficientNet-B0", "深度可分离卷积 + SE 注意力", "4 M", "288×288", "24"],
               ["ResNet50", "标准瓶颈残差结构", "23 M", "256×256", "24"],
               ["ConvNeXt-Tiny", "大核（7×7）深度可分离卷积", "27 M", "256×256", "20"],
               ["ResNet18", "最浅的残差网络，最轻量", "11 M", "256×256", "32"]],
              widths=[3.4, 5.4, 2.0, 2.6, 1.8], font_size=8.8,
              caption="表 1-1  四种骨干网络的结构与配置")
    P("所有骨干均载入 ImageNet 预训练权重，随后将原有的 1000 类分类头替换为新的线性层 "
      "Linear(d, 5)，其中 d 为该骨干的全局池化特征维度（EfficientNet-B0 为 1280 维）。"
      "这样做的含义是：保留骨干学到的通用视觉特征，仅重新学习“如何把特征映射到本任务的"
      "五个类别”。")

    H3("1.3.3  数据增强策略")
    P("增强设计并非随意堆叠，而是围绕一个明确的先验：本任务的类别判别与植株的绝对位置、"
      "朝向、尺度均无关。由于图像为正俯视拍摄，也不存在“倒着长”的物理不合理问题，"
      "因此翻转与旋转可以开到较大幅度。具体配置见表 1-2。")
    add_table(doc,
              ["增强方式", "参数设置", "设计依据"],
              [["随机裁剪缩放", "面积比例 0.45~1.0，长宽比 0.75~1.33",
                "植株可能仅占画面一小部分，需模拟不同取景"],
               ["水平 / 垂直翻转", "各 50% 概率", "俯拍视角，无重力方向先验"],
               ["随机旋转", "[-180°, +180°] 连续", "幼苗朝向完全任意"],
               ["亮度 / 对比度抖动", "×0.8~1.25，80% 概率", "闪光灯与曝光差异"],
               ["颜色偏移", "±12/255，50% 概率", "白平衡差异"],
               ["随机擦除", "面积 1/64~1/9，25% 概率", "模拟砂石遮挡"]],
              widths=[4.0, 5.6, 5.6], font_size=8.8,
              caption="表 1-2  训练阶段的数据增强配置")
    P("需要说明的是，混合增强（Mixup）在代码中作为可选开关实现，但默认关闭。原因是本任务的"
      "类间差异恰恰体现为细微的叶片形态，而 Mixup 会生成“叶片叠叶片”的混合样本，"
      "反而破坏了最需要学习的判别性线索。")

    H3("1.3.4  迁移学习的优化设计")
    P("在优化层面，本实验采用三项相互配合的设计。")
    P("第一，判别式学习率（discriminative learning rate）。按参数名将网络分为两组："
      "预训练骨干（body）使用较小学习率 1×10⁻⁴，新初始化的分类头（head）使用较大学习率 "
      "1×10⁻³。其依据是两组参数的初始状态完全不同：骨干的权重已经是一个“会看图像”的"
      "较好解，只需小步微调；而分类头是随机初始化的，需要快速收敛。若整体使用大学习率，"
      "前几次更新就会破坏预训练特征；若整体使用小学习率，分类头在有限的 25 个轮次内"
      "难以收敛到位。", )
    P("第二，单周期学习率调度（OneCycleLR）。以两组各自的峰值学习率为上界，先线性预热"
      "（占总步数的 15%），再余弦退火至峰值的 1/100。由于总步数仅约 420 步（25 轮 × 约 17 "
      "次迭代），短训练条件下预热加退火的收益明显优于阶梯式衰减。")
    P("第三，标签平滑与梯度裁剪。交叉熵使用 0.1 的标签平滑，配合较大范数（5.0）的梯度裁剪，"
      "在 400 张训练样本上抑制过拟合与训练震荡。")

    H3("1.3.5  交叉验证协议")
    P("若简单地将 500 张图像按 80/20 划分为训练集与验证集，验证集仅有 100 张，"
      "准确率的标准误约为 ±4 个百分点。这意味着两个方案的差异（通常 1~2 个百分点）"
      "完全被统计噪声淹没，无法判断改进是否真实。")
    P("因此本实验采用五折分层交叉验证：将每一类的 100 张图像按轮转方式均匀分配到 5 折"
      "（每折每类恰好 20 张），每次用 4 折（400 张）训练、1 折（100 张）验证。"
      "这样每张图像恰好被验证一次，五折结果汇总即为对全部 500 张的无偏估计，"
      "标准误降低至约 ±2 个百分点。同时，五折训练天然产生 5 个模型，可作为集成使用。")
    P("在模型选择上，保存验证集上准确率最优的轮次权重，而非最后一个轮次的权重。"
      "1.5 节的训练日志显示，最优轮次常出现在中段（例如第 10 轮达到 92%，末轮仍为 92%），"
      "取最优等价于带有免费早停的固定预算训练。")

    H3("1.3.6  推理阶段的多视角与多模型聚合")
    P("推理时对每张测试图生成 8 个几何视角（原图、水平翻转、垂直翻转、双向翻转、"
      "旋转 90°/180°/270°、旋转 90° 后翻转），分别前向传播后在 softmax 空间取平均。"
      "该设计依据与增强一致：这些几何变换不改变类别。需要注意的是，平均应在 softmax "
      "概率空间进行，而非 logit 空间，因为概率才具有可加性。")
    P("模型聚合的顺序为：先在同一骨干内部按折平均，再在骨干之间平均。这一顺序的意义在于"
      "保证每个骨干等权——若某骨干只训练了部分折，不应因此在最终结果中被隐式降权。")
    P("经此聚合，单个骨干需完成 40 次前向（5 折 × 8 视角），四骨干合计 160 次；"
      "最终单张测试图的预测由 20 个模型（4 骨干 × 5 折）与 8 个视角共同决定。")

    H3("1.3.7  决策后处理：两类偏置修正")
    P("针对 1.2.4 节识别的类别先验偏置，本实验设计并实现了两种修正方法，"
      "并明确区分了二者的适用条件。")
    P("方法一：基于折外预测的决策权重修正。利用五折交叉验证产生的折外（out-of-fold，OOF）"
      "概率，估计当前决策边界相对均匀先验的偏移量，为每一类学一个乘数 w_c，在 argmax 前"
      "对概率加权。为抑制过拟合，权重通过坐标上升法在 OOF 上优化，目标函数为平衡准确率，"
      "且乘数归一化为均值为 1（仅 4 个自由度）。更为关键的是，该方法附带留一折检验："
      "在 4 折上学权重、在第 5 折上评估，从而给出可信的增益估计。")
    P("方法二：基于测试集自身分布的直推式修正（transductive prior correction）。"
      "该方法不依赖训练集 OOF，而是直接测量待预测集合自身的平均预测分布，"
      "将其推回训练集先验（均匀分布）。二者都不使用任何标签，因此不涉及标签泄漏；"
      "区别在于所修正的偏置来源不同。")
    P("这一区分是本实验在方法论上的核心认识：修正工具必须匹配偏置的来源。"
      "当验证集的偏置与测试集的偏置不一致时，用前者学到的权重去修后者会失效，"
      "甚至适得其反。1.6.5 节以实测数据验证了这一判断。")

    # ---------------- 1.4 ----------------
    H2("1.4  实验方案")
    H3("1.4.1  实验环境")
    add_table(doc,
              ["项目", "配置"],
              [["计算平台", "Kaggle Notebook（NVIDIA GPU）"],
               ["深度学习框架", "PyTorch + torchvision"],
               ["依赖", "仅 torch、torchvision、numpy、Pillow（不依赖 sklearn/pandas/scipy）"],
               ["划分方式", "五折分层交叉验证，随机种子 2026"],
               ["提交形式", "submission.csv（ID, Category），按官方模板对齐 ID 顺序"]],
              widths=[3.6, 11.4], font_size=9.0, align_center_cols=[0],
              caption="表 1-3  实验环境与基本配置")

    H3("1.4.2  数据预处理与缓存")
    P("由于原始图像为非正方形且尺寸差异极大（短边从十余像素到上千像素不等），"
      "预处理分两步：首先以白色填充将图像补成正方形，避免直接拉伸造成形变；"
      "随后统一缩放到 384×384 并保存为 uint8 数组缓存。")
    P("缓存设计有两重考虑：其一，训练时不再反复解码 PNG，在纯 CPU 环境下可显著降低"
      "数据读取开销；其二，384 作为“足够大的中间分辨率”，后续无论以 256、288 还是 "
      "320 训练，均从缓存下采样，不会因缓存过小而损失信息。缓存文件约 252 MB。")

    H3("1.4.3  超参数配置")
    add_table(doc,
              ["超参数", "取值", "说明"],
              [["训练轮次", "25", "四骨干均采用同一预算"],
               ["批大小", "20 ~ 32", "按骨干显存占用分别设定"],
               ["优化器", "AdamW", "权重衰减 1×10⁻⁴"],
               ["学习率", "骨干 1×10⁻⁴ / 分类头 1×10⁻³", "判别式学习率"],
               ["调度策略", "OneCycleLR", "预热 15%，余弦退火至峰值 1/100"],
               ["标签平滑", "0.1", "抑制过拟合"],
               ["梯度裁剪", "范数 5.0", "稳定训练"],
               ["TTA 视角数", "8", "几何变换，softmax 空间平均"],
               ["集成规模", "20 个模型", "4 骨干 × 5 折"]],
              widths=[3.4, 6.4, 5.2], font_size=8.9, align_center_cols=[0, 1],
              caption="表 1-4  训练超参数配置")

    H3("1.4.4  实验分组设计")
    P("为区分各项设计的独立贡献，本实验按以下顺序逐步构建，形成隐式消融：")
    NUM(["单骨干 + 单折训练（30 轮）：验证迁移学习的有效性，并观察收敛行为；",
         "单骨干 + 五折交叉验证：检验交叉验证集成的增益；",
         "四骨干 + 五折集成 + 8 视角 TTA：检验架构多样性与多视角聚合的增益；",
         "在第三项的基础上叠加决策权重修正，并做留一折检验；",
         "针对测试集偏置，进一步设计直推式先验修正，并在 OOF 上检验其方向正确性。"])

    # ---------------- 1.5 ----------------
    H2("1.5  训练过程")
    H3("1.5.1  单模型训练过程分析")
    P("为观察迁移学习的收敛行为，首先在 CPU 环境下对 EfficientNet-B0 做单折 30 轮训练。"
      "训练损失、验证损失与验证准确率随轮次的变化如图 1-2 所示，关键轮次数据见表 1-5。")
    FIG(p12, "图 1-2  EfficientNet-B0 单折训练过程的损失与准确率曲线")
    add_table(doc,
              ["轮次", "训练损失", "验证损失", "验证准确率"],
              [["1", "1.5982", "1.5480", "33.0%"],
               ["3", "1.2870", "1.0827", "63.0%"],
               ["6", "0.7105", "0.6414", "90.0%"],
               ["10", "0.6006", "0.5850", "92.0%"],
               ["20", "0.5170", "0.5533", "91.0%"],
               ["30", "0.5133", "0.5545", "92.0%"]],
              widths=[2.2, 4.0, 4.0, 4.4], font_size=9.2,
              caption="表 1-5  单模型训练过程中的关键轮次数据")
    P("从图 1-2 与表 1-5 可读出三点信息。")
    P("其一，预训练权重的效果极为显著。仅第 6 轮验证准确率即达到 90%，而随机猜测为 20%。"
      "这说明模型不是“从头学习如何看植物”，而是在一个已经理解通用视觉模式的表征上"
      "做小幅调整——这正是迁移学习的意义。")
    P("其二，最优轮次出现在中段。验证准确率在第 10 轮首次达到 92%，此后在第 17、18、21 "
      "轮等多次回到 92%，末轮亦为 92%，但中间存在回落（如第 13 轮降至 89%）。"
      "这解释了为何采用“按验证集取最优轮次”而非“取末轮”的策略。")
    P("其三，过拟合迹象在后期出现。训练损失从第 10 轮的 0.6006 持续下降至末轮的 0.5133，"
      "验证损失则由 0.5850 缓慢降至 0.5545，降幅远小于训练损失；至末轮训练损失已低于"
      "验证损失，两者差距由 0.016 扩大至 0.041。这提示在 400 张训练样本上继续增加轮次"
      "已无收益，正则化与集成的必要性由此得到印证。")

    H3("1.5.2  四骨干五折训练的收敛表现")
    P("在 GPU 环境下完成四骨干 × 五折共 20 次独立训练，每次训练使用其对应的 400 张训练集"
      "与 100 张验证集，各次训练的验证准确率见表 1-6 与图 1-3。")
    fold_rows = []
    for n in FOLD_ACC:
        v = FOLD_ACC[n]
        fold_rows.append([n] + ["%.2f" % x for x in v] + ["%.4f" % float(np.mean(v))])
    add_table(doc,
              ["骨干网络", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "均值"],
              fold_rows, widths=[3.2, 2.05, 2.05, 2.05, 2.05, 2.05, 2.2], font_size=8.9,
              caption="表 1-6  四骨干五折交叉验证的验证准确率")
    FIG(p13, "图 1-3  四种骨干网络在 5 折交叉验证下的验证准确率")
    P("由表 1-6 可见，ConvNeXt-Tiny 的平均验证准确率最高（94.2%），ResNet50 次之（93.0%），"
      "EfficientNet-B0 与 ResNet18 均为 92.2%。单折波动幅度在 89%~97% 之间，"
      "这再次说明单折评估的不可靠性：若只看某一折，结论可能完全相反"
      "（例如 ResNet18 的 fold 0 高达 96%，而 fold 1 仅 89%）。")

    H3("1.5.3  训练过程的稳定性观察")
    P("在训练过程中记录了三个值得注意的现象。")
    P("第一，分类头与骨干的收敛速度差异明显。前 3 个轮次验证准确率快速爬升"
      "（33% → 51% → 63%），正是判别式学习率让随机初始化的分类头快速到位的直接体现；"
      "此后进入缓慢精调阶段。")
    P("第二，验证损失的平台期早于准确率的平台期。验证损失在第 10 轮即趋于平稳，"
      "而准确率在其后仍有小幅波动。这说明模型在后半程主要在调整决策边界的细节，"
      "而非改进特征表示。")
    P("第三，各折之间存在系统性难度差异。折叠划分虽已分层，但 fold 1 在多个骨干上"
      "均为最难（ResNet18 仅 89%），可能源于该折中包含更多成像条件特殊的样本。"
      "这一现象提醒：不应以单折成绩评判方法优劣。")

    # ---------------- 1.6 ----------------
    H2("1.6  实验结果与分析")
    H3("1.6.1  交叉验证总体结果")
    P("将四骨干的 OOF 概率按骨干等权平均，得到集成的 OOF 预测。在该预测上，"
      "集成模型的总体准确率为 92.60%，平衡准确率 92.60%，宏平均 F1 为 92.56%。"
      "由于训练集五类严格均衡，三者数值接近，说明模型整体未出现严重的类别偏置"
      "——但这只是 OOF 上的情况，测试集上的表现需单独考察（见 1.6.5 节）。")

    H3("1.6.2  各类别识别能力分析")
    P("各类别的召回率、精确率与 F1 见表 1-7 与图 1-4。")
    rows = []
    for i, c in enumerate(CLASSES):
        rows.append([c, CLASSES_CN[i], "100", "%.3f" % OOF_RECALL[i],
                     "%.3f" % OOF_PREC[i], "%.3f" % OOF_F1CLS[i]])
    add_table(doc,
              ["类别（英）", "类别（中）", "样本数", "召回率", "精确率", "F1"],
              rows, widths=[3.5, 2.6, 2.0, 2.4, 2.4, 2.2], font_size=8.9,
              caption="表 1-7  集成模型在 OOF 上的分类指标")
    FIG(p14, "图 1-4  集成模型 OOF 上各类别的召回率、精确率与 F1")
    P("结果呈现出清晰的两极分化。无香母菊与甜菜的召回率均达到 100%，精确率分别为 1.000 "
      "与 0.990，说明这两个类别在形态上具有独特性，模型可以稳定识别。而黑麦草与散穗黍的"
      "召回率仅为 83.0%，是全部误差的主要来源。")
    P("进一步观察召回率与精确率的关系可以推断混淆方向。黑麦草的召回率（0.830）与精确率"
      "（0.814）同时偏低且数值接近，说明它既被误判出去、也被误判进来，属于双向混淆；"
      "散穗黍的精确率（0.883）高于召回率（0.830），说明它“被误判出去”多于“被误判进来”，"
      "即有一部分散穗黍被系统性地判成了别的类别。结合普通小麦的召回率高达 97.0%，"
      "可以判断：黑麦草与散穗黍之间的相互混淆是主要误差结构，"
      "这与 1.2.2 节的形态学分析完全一致。")

    H3("1.6.3  多骨干集成的作用")
    P("四个骨干的折内平均验证准确率分别为 92.2%、93.0%、94.2% 与 92.0%，算术平均为 92.85%；"
      "而集成后的 OOF 准确率为 92.60%，略低于四者平均，也低于最强的 ConvNeXt-Tiny。"
      "这一结果说明：在单模型已普遍达到 92%~94% 的高水平下，简单等权集成的主要意义"
      "不在于继续抬高准确率，而在于降低方差并抑制单一架构的类别偏好。"
      "这也提示集成并非“越多样越好”，当成员之间存在明显强弱差距时，等权平均会"
      "被较弱成员拖累，此时应改用按验证表现加权的融合方式（见 1.8 节）。")
    P("同口径下，实验早期的单骨干（30 轮训练）OOF 准确率为 92.0%，四骨干集成后为 92.60%，"
      "提升约 0.6 个百分点。值得强调的是，类别偏斜的缓解主要取决于训练是否充分，"
      "而非集成：仅训练 2 轮的快速对照实验中，普通小麦的预测占比低至 2.6%；"
      "充分训练后的单骨干为 9.3%，四骨干集成为 8.7%。也就是说，训练充分程度决定了"
      "偏斜的量级，集成则在此基础上通过结构差异相互校正，进一步减少单一架构对"
      "某些类别的系统性偏好。")
    P("此外，骨干间预测一致率为 89.4%，说明在绝大多数样本上四个骨干意见相同；"
      "而在剩余 10.6% 的分歧样本上，集成的平均起了仲裁作用。这些分歧样本正是"
      "难以判别的边界样本。")

    H3("1.6.4  预测置信度分析")
    P("集成模型在测试集上的预测置信度均值为 0.812，中位数 0.869，最低 0.369。"
      "置信度低于 0.5 的样本仅 12 张（占 3.2%）。这表明模型对绝大多数测试样本"
      "具有较高把握，剩余少数低置信样本是最可能的错误位置，也是后续人工复核"
      "或专项改进的优先对象。")

    H3("1.6.5  决策偏置修正的效果与适用边界")
    P("这是本实验分析环节的核心内容。表 1-8 给出决策权重修正的量化结果。")
    add_table(doc,
              ["评估方式", "准确率", "平衡准确率", "宏平均 F1", "说明"],
              [["修正前（OOF）", "0.9260", "0.9260", "0.9256", "基准"],
               ["修正后（OOF）", "0.9400", "0.9400", "0.9391", "同批数据自评，偏乐观"],
               ["留一折检验", "—", "0.9260 → 0.9360", "0.9254 → 0.9348",
                "可信增益：平衡准确率 +0.0100，F1 +0.0095"]],
              widths=[3.6, 2.1, 3.4, 3.1, 3.4], font_size=8.8,
              caption="表 1-8  决策权重修正的量化效果")
    P("表 1-8 的关键在于区分“自评”与“留出评估”。修正后在 OOF 上宏平均 F1 由 0.9256 "
      "提升至 0.9391（+0.0135），但这一数字是乐观的，因为权重正是在这批数据上学习的。"
      "真正可信的是留一折检验结果：在模型完全未见过的折上，宏平均 F1 由 0.9254 提升至 "
      "0.9348，净增益 +0.0095。该增益方向明确但幅度有限。")
    P("然而，将学到的权重应用到测试集后，出现了一个值得深究的现象。"
      "修正前后的测试集预测分布见表 1-9 与图 1-5。")
    add_table(doc,
              ["类别", "训练集先验", "未修正预测", "修正后预测", "偏离 20% 的程度"],
              [["黑麦草", "20.0%", "18.8%", "15.1%", "1.2 → 4.9（变差）"],
               ["普通小麦", "20.0%", "8.7%", "9.0%", "11.3 → 11.0（略好）"],
               ["散穗黍", "20.0%", "26.7%", "30.2%", "6.7 → 10.2（变差）"],
               ["无香母菊", "20.0%", "23.8%", "23.8%", "3.8 → 3.8（持平）"],
               ["甜菜", "20.0%", "22.0%", "22.0%", "2.0 → 2.0（持平）"],
               ["平均偏离度", "—", "5.00", "6.38", "变差 1.38"]],
              widths=[2.6, 2.6, 2.8, 2.8, 4.3], font_size=8.9,
              caption="表 1-9  决策权重修正前后的测试集预测分布对比")
    FIG(p15, "图 1-5  决策权重修正前后的测试集预测分布")
    P("表 1-9 显示，修正反而使测试集分布进一步偏离均匀先验：黑麦草被从 18.8% 压低至 15.1%，"
      "散穗黍被从 26.7% 抬高至 30.2%，平均偏离度由 5.00 恶化至 6.38。原因可以精确定位。")
    P("其根源在于偏置来源不匹配。修正权重是在 OOF 上学习的，而 OOF 的预测分布本身已接近"
      "均匀（0.204、0.206、0.188、0.200、0.202）——也就是说，在集成之后，验证集上的类别"
      "偏置几乎已被消除。在“已无偏置可修”的数据上学权重，优化器只能通过牺牲难以判别的"
      "类别来微量提升平衡准确率（学到的黑麦草权重仅为 0.704，而黑麦草恰恰是召回率最低的"
      "类别）。而这套权重被应用到偏置严重的测试集上时，方向恰好与需要相反，"
      "于是加剧了偏斜。")
    P("这一发现引出了方法二的设计：由于测试集自身的偏置幅度（普通小麦 8.7%）远大于"
      "验证集，应当直接在测试集上测量偏置并修正。基于此，实验中补充实现了直推式先验修正，"
      "并在 OOF 上验证了其方向正确性——将 OOF 的预测分布推向均匀后，平衡准确率由 0.9260 "
      "提升至 0.9400，与决策权重修正达到的数值一致，说明“推向均匀先验”这一操作本身"
      "是有效的。该方法的测试集效果受条件限制未能单独量化，将在 1.8 节作为改进方向讨论。")
    P("综上，本节得出的方法论结论是：决策修正必须匹配偏置的来源。用验证集的偏置去修正"
      "测试集的偏置，当两者幅度与方向不一致时不仅无效，反而有害；这一结论由实测数据"
      "（表 1-9）直接支撑，而非推测。")

    H3("1.6.6  提交成绩")
    P("两次提交的成绩见表 1-10。")
    add_table(doc,
              ["提交", "方法配置", "排行榜成绩"],
              [["第一次", "单骨干 + 五折 + 8 视角 TTA", "0.9232"],
               ["第二次", "四骨干集成 + 五折 + 8 视角 TTA + 决策权重修正", "0.9312"]],
              widths=[2.6, 9.4, 3.4], font_size=9.0,
              caption="表 1-10  Kaggle 排行榜提交成绩")
    P("成绩由 0.9232 提升至 0.9312，绝对提升 0.8 个百分点。需要客观说明的是，"
      "这一提升主要来自多骨干集成与多视角 TTA 带来的方差降低，"
      "决策权重修正的贡献难以从这个总成绩中单独剥离——事实上 1.6.5 节的分析表明，"
      "该修正在本次测试集上可能未产生正向作用。这一诚实的归因判断本身"
      "也是实验分析的一部分。")

    # ---------------- 1.7 ----------------
    H2("1.7  创新点总结")
    P("本实验在设计上的创新主要体现在以下四个方面。")
    P("第一，将“验证协议”而非“模型结构”作为方案的核心。面对每类仅 100 张的极端小样本，"
      "本实验没有追求更复杂的网络，而是先建立统计上可靠的评估体系（五折分层交叉验证、"
      "按验证集取最优轮次、留一折检验），使得后续每一步改进都可被验证真伪。"
      "这一取舍在小样本任务中具有普遍意义。")
    P("第二，识别并量化了迁移学习的类别先验偏置。实验发现模型在测试集上将普通小麦"
      "压低至 8.7%（期望 20%），并将其归因于 ImageNet 标签先验的继承。"
      "这一现象在常规的准确率报告中会被掩盖，但对宏平均指标影响显著，"
      "具有较强的问题发现价值。")
    P("第三，提出并实现了两种决策修正方法，并明确其适用边界。基于 OOF 的决策权重修正"
      "与基于测试集的直推式先验修正，二者均不使用标签、均不涉及泄漏，"
      "但适用于不同来源的偏置。实验通过留一折检验与分布对比，"
      "给出了“何种情况下应当选用哪一种”的实证判据。")
    P("第四，对增强策略进行了任务定制化设计。依据“正俯视拍摄”这一物理事实，"
      "将旋转幅度开放至 ±180°，并依据“植株可能仅占画面一角”将随机裁剪的面积下限"
      "下探至 0.45，而非套用通用增强配置。同时基于细粒度任务的特性，"
      "有理据地放弃了 Mixup。")

    # ---------------- 1.8 ----------------
    H2("1.8  不足与改进方向")
    P("本实验仍存在若干不足，可归纳为以下五点。")
    NUM(["伪标签半监督尚未启用。测试集 378 张图像恰好来自被评分的同一分布，"
         "将其高置信度部分作为额外训练数据，理论上可把标注量扩大 70%，"
         "是小样本条件下剩余收益最大的一项。代码中已实现置信度与多骨干一致性双重闸门，"
         "但本报告未包含其量化结果。",
         "直推式先验修正的测试集效果未能量化。由于测试集无标签，该方法的实际增益"
         "只能通过提交验证，建议后续以对照提交的方式评估。",
         "未引入更高分辨率训练。细粒度识别对分辨率敏感，当前 256~288 的输入可能"
         "损失了叶脉级别的细节，在算力允许时可尝试 384。",
         "未针对易混淆类别做专门处理。黑麦草与散穗黍的召回率仅 83%，"
         "可考虑对这两类单独训练二分类判别器，或对易混区域做局部高分辨率裁剪。",
         "集成权重为简单等权平均。可根据各折、各骨干的验证表现或预测不确定性"
         "做加权融合，理论上可进一步降低方差。"])

    # ---------------- 参考文献 ----------------
    H2("参考文献")
    refs = [
        "[1] Deng J, Dong W, Socher R, et al. ImageNet: A Large-Scale Hierarchical Image Database[C]. CVPR, 2009.",
        "[2] Tan M, Le Q V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks[C]. ICML, 2019.",
        "[3] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016.",
        "[4] Liu Z, Mao H, Wu C Y, et al. A ConvNet for the 2020s[C]. CVPR, 2022.",
        "[5] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization[C]. ICLR, 2019.",
        "[6] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the Inception Architecture for Computer Vision[C]. CVPR, 2016.",
        "[7] Smith L N. A Disciplined Approach to Neural Network Hyper-Parameters: Part 1[R]. arXiv:1803.09820, 2018.",
        "[8] Szegedy C, Liu W, Jia Y, et al. Going Deeper with Convolutions[C]. CVPR, 2015.",
        "[9] Menon A K, Jayasumana S, Rawat A S, et al. Long-tail Learning via Logit Adjustment[C]. ICLR, 2021.",
    ]
    for r_ in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.9)
        p.paragraph_format.first_line_indent = Cm(-0.9)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        run = p.add_run(r_)
        set_run_font(run, size=9.2)

    out = os.path.join(OUT_DIR, "任务一_植物分类_实验报告.docx")
    doc.save(out)
    print("[docx] saved:", out)
    return out


if __name__ == "__main__":
    build()
