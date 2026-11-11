# -*- coding: utf-8 -*-
"""
生成课程设计实验报告（任务一：植物分类）—— 精简版，目标 6~8 页。

与详细版的差别：
  * 图 5 -> 3（保留技术路线、训练曲线、预测分布对比），
    各折准确率与各类指标改由表格承载；
  * 表 11 -> 5（骨干配置、实验配置、各折准确率、各类指标、修正效果）；
  * 正文约 6400 字 -> 约 4300 字，合并 1.4/1.5 的冗余小节；
  * 收紧标题与图题间距，参考文献改单倍行距小字号。

所有数值仍取自本次实验的真实产物，未作虚构。
"""
import os
import sys

# 复用详细版的绘图与排版工具，避免两份代码走样
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_report import (  # noqa: E402
    CLASSES, CLASSES_CN, OOF_RECALL, OOF_PREC, OOF_F1CLS, FOLD_ACC,
    setup_font, fig_pipeline, fig_curve, fig_dist,
    set_run_font, add_page_number_footer, add_caption,
    shade_cell, set_table_borders, set_table_fixed_layout,
)

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
OUT_DIR = os.path.join(PROJECT, "report")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

USABLE = 15.8


def add_table(doc, header, rows, widths, font_size=8.8, caption=None,
              align_center_cols=None):
    """精简版表格：固定列宽、同步 tblGrid、居中。"""
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
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(h))
        set_run_font(r, size=font_size, bold=True)
        shade_cell(hdr[i], "DCE6F1")

    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER
                           if (align_center_cols is None or i in align_center_cols)
                           else WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(str(v))
            set_run_font(r, size=font_size)

    for r_ in table.rows:
        for i, w in enumerate(widths):
            r_.cells[i].width = Cm(w)
    grid = table._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(Cm(w).twips)))
    tblW = table._tbl.tblPr.find(qn("w:tblW"))
    if tblW is not None:
        tblW.set(qn("w:type"), "dxa")
        tblW.set(qn("w:w"), str(int(Cm(sum(widths)).twips)))
    assert sum(widths) <= USABLE + 1e-6, "表格超宽: %.2f" % sum(widths)


def build():
    setup_font()
    p11 = fig_pipeline()
    p12 = fig_curve()
    p15 = fig_dist()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.6)
    sec.top_margin = Cm(2.3)
    sec.bottom_margin = Cm(2.1)
    add_page_number_footer(sec)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(0)

    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
        st = doc.styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        st.paragraph_format.space_before = Pt(7 if name == "Heading 1" else 6)
        st.paragraph_format.space_after = Pt(3)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    def H1(t):
        h = doc.add_heading(level=1)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=15, bold=True)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def H2(t):
        h = doc.add_heading(level=2)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=12.5, bold=True)

    def H3(t):
        h = doc.add_heading(level=3)
        r = h.add_run(t)
        set_run_font(r, ea_font="黑体", size=11, bold=True)

    def P(text, indent=True, size=10.5, after=0):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74) if indent else Cm(0)
        p.paragraph_format.space_after = Pt(after)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(text)
        set_run_font(r, size=size)
        return p

    def NUM(items, size=10.5):
        for it in items:
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.left_indent = Cm(0.85)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
            r = p.add_run(it)
            set_run_font(r, size=size)

    def FIG(path, caption, width_cm):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(0)
        p.add_run().add_picture(path, width=Cm(width_cm))
        add_caption(doc, caption, kind="figure")

    # ===================================================================== #
    #  题头（不单独占页）
    # ===================================================================== #
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(0)
    t.paragraph_format.space_after = Pt(2)
    r = t.add_run("机器学习课程设计实验报告")
    set_run_font(r, ea_font="黑体", size=18, bold=True)

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st.paragraph_format.space_after = Pt(2)
    r = st.add_run("—— 任务一：植物分类 ——")
    set_run_font(r, ea_font="黑体", size=12, bold=True)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.paragraph_format.space_after = Pt(6)
    r = info.add_run("五分类图像识别　|　训练 500 张（五类各 100）　|　测试 378 张　|　最佳提交成绩 0.9312")
    set_run_font(r, size=9)

    # ===================================================================== #
    H1("第 1 章  植物分类")

    H2("1.1  任务背景与问题描述")
    P("本任务对植物幼苗图像作五分类：黑麦草（Black-grass）、普通小麦（Common wheat）、"
      "散穗黍（Loose Silky-bent）、无香母菊（Scentless Mayweed）与甜菜（Sugar beet）。"
      "训练集 500 张（五类各 100 张），测试集 378 张无标注，提交两列 CSV（ID、Category），"
      "按排行榜准确率排名。")
    P("难点不在网络搭建，而在数据条件：每类仅 100 张样本，而禾本科三物种在幼苗期的差异"
      "仅体现在叶宽、叶脉走向与叶鞘颜色上。因此本实验不追求更复杂的结构，而是以"
      "“榨取 ImageNet 预训练特征”为主线，并优先建立统计上可靠的验证协议，"
      "使每一步改进都能被证伪。")

    H2("1.2  问题分析")
    P("其一，小样本约束。五类共 500 张；若从随机初始化训练 EfficientNet-B0（约 400 万参数），"
      "参数量约为样本量的 8000 倍，模型有无数种解可完美拟合这 500 张，却无法保证对第 501 张"
      "的判断。实验亦印证：仅训练 6 轮，验证准确率即达 90%，而随机猜测为 20%。")
    P("其二，类内变异大于类间差异。同一物种的图像在植株位置与尺度（有的仅占画面一角）、"
      "朝向（俯拍下完全任意）、光照（闪光灯与散射光并存）与背景（砂石、标签纸、育苗盘）上"
      "差异极大，这是增强策略设计的直接依据。")
    P("其三，类别先验偏置（本实验的关键发现）。训练集五类严格均衡，故测试集真实分布应接近"
      "每类 20%；但集成模型在测试集上的原始预测为 18.8%、8.7%、26.7%、23.8%、22.0%，"
      "普通小麦被压至 8.7%。该偏置并非特征质量问题，而是 ImageNet 标签先验的继承"
      "（禾本科杂草样本远多于栽培作物），表现为决策边界整体偏移，在 macro-F1 上被显著放大，"
      "且无法靠增加轮次或更换骨干消除。这是本实验决策后处理的出发点。")

    H2("1.3  模型与方法设计")
    H3("1.3.1  总体技术路线")
    P("整体流程为：预处理与缓存 → 训练随机增强 → 迁移学习与五折交叉验证 → "
      "多视角多模型聚合 → 决策后处理，如图 1-1 所示。")
    FIG(p11, "图 1-1  总体技术路线", 11.6)

    H3("1.3.2  骨干网络与分类头")
    P("选用四种归纳偏置差异明显的骨干：EfficientNet-B0（深度可分离卷积 + SE 注意力）、"
      "ResNet50（瓶颈残差）、ConvNeXt-Tiny（大核深度可分离）、ResNet18（最浅）。"
      "依据是集成学习中的误差去相关原理——若各模型错在同一批图像上，平均无效；"
      "结构差异越大，平均才越能降低方差。各骨干均载入 ImageNet 预训练权重，"
      "并将分类头替换为 Linear(d, 5)（表 1-1）。")
    add_table(doc,
              ["骨干网络", "结构特点", "参数量", "输入尺寸", "批大小"],
              [["EfficientNet-B0", "深度可分离卷积 + SE 注意力", "4 M", "288×288", "24"],
               ["ResNet50", "标准瓶颈残差结构", "23 M", "256×256", "24"],
               ["ConvNeXt-Tiny", "大核（7×7）深度可分离卷积", "27 M", "256×256", "20"],
               ["ResNet18", "最浅的残差网络，最轻量", "11 M", "256×256", "32"]],
              widths=[3.5, 5.6, 2.0, 2.6, 1.9], caption="表 1-1  四种骨干网络的结构与配置")

    H3("1.3.3  数据增强策略")
    P("增强围绕“类别判别与绝对位置、朝向、尺度均无关”这一先验设计。因图像为正俯视拍摄，"
      "不存在“倒着长”的物理不合理，故双向翻转各 50%、旋转开放至 ±180°；"
      "随机裁剪面积比例 0.45~1.0 以模拟植株仅占画面一角的取景；另加亮度与对比度抖动"
      "（×0.8~1.25）、±12/255 颜色偏移、以及 25% 概率的随机擦除（面积 1/64~1/9）"
      "以模拟光照差异与砂石遮挡。Mixup 虽已实现但默认关闭：本任务的类间差异恰为细微的"
      "叶片形态，混合样本会破坏最需学习的判别线索。")

    H3("1.3.4  优化设计与验证协议")
    P("优化上采用判别式学习率：预训练骨干用小学习率 1×10⁻⁴（其权重已属较好解，只需微调），"
      "新分类头用 1×10⁻³（随机初始化，需快速收敛）。若整体用大学习率，前几次更新即会破坏"
      "预训练特征；若整体用小学习率，分类头在 25 轮内难以到位。调度采用 OneCycleLR"
      "（预热 15%，余弦退火至峰值 1/100），并配合标签平滑 0.1 与范数 5.0 的梯度裁剪。")
    P("验证采用五折分层交叉验证：每类 100 张按轮转均分至 5 折，每折每类恰 20 张，"
      "每次以 4 折（400 张）训练、1 折（100 张）验证。若简单地按 80/20 划分，验证集仅 100 张，"
      "准确率标准误约 ±4 个百分点，足以淹没 1~2 个点的方案差异；五折使每张图像恰被验证一次，"
      "标准误降至约 ±2 点，并附带产生 5 个模型。训练中保存验证准确率最优轮次的权重而非末轮，"
      "等价于带有免费早停的固定预算训练。")

    H3("1.3.5  推理聚合与决策后处理")
    P("推理时对每张测试图生成 8 个几何视角（原图、双向翻转、旋转 90°/180°/270° 等），"
      "分别前向后在 softmax 概率空间平均（概率才具可加性）。聚合顺序为先骨干内按折平均、"
      "再骨干间平均，以保证骨干等权。单个骨干需 40 次前向（5 折 × 8 视角），四骨干合计 160 次；"
      "单张测试图的预测由 20 个模型（4 骨干 × 5 折）与 8 个视角共同决定。")
    P("针对 1.2 节的偏置，实现两种均不使用标签的修正。方法一为 OOF 决策权重修正："
      "在五折折外概率上用坐标上升法为每类学一个乘数（归一化为均值 1，仅 4 个自由度），"
      "目标函数为平衡准确率，并附留一折检验——在 4 折上学、在第 5 折上评估，以给出可信增益。"
      "方法二为直推式先验修正：直接测量待预测集合自身的平均预测分布，将其推回均匀先验。"
      "二者的关键区别在于所修偏置的来源不同，这一点构成 1.6.3 节的核心结论。")

    H2("1.4  实验方案")
    P("预处理分两步：先以白色填充将非正方形原图补成正方形（避免直接拉伸形变），"
      "再统一缩放至 384×384 并缓存为 uint8 数组（约 252 MB）。缓存既避免训练时反复解码 PNG，"
      "又保证后续无论以 256/288/320 训练，均从同一中间分辨率下采样而不损失信息。"
      "实验环境与超参数见表 1-2。")
    add_table(doc,
              ["项目", "配置"],
              [["计算平台 / 框架", "Kaggle Notebook（NVIDIA GPU）/ PyTorch + torchvision"],
               ["依赖", "仅 torch、torchvision、numpy、Pillow（不依赖 sklearn、pandas、scipy）"],
               ["输入分辨率 / 缓存", "288 或 256（按骨干）/ 384×384 缓存"],
               ["训练轮次 / 优化器", "25 轮 / AdamW（权重衰减 1×10⁻⁴）"],
               ["学习率", "骨干 1×10⁻⁴，分类头 1×10⁻³（判别式）"],
               ["调度 / 正则化", "OneCycleLR（预热 15%）/ 标签平滑 0.1、梯度裁剪 5.0"],
               ["推理 / 集成规模", "8 视角 TTA，softmax 平均 / 20 个模型"]],
              widths=[3.6, 11.9], align_center_cols=[0],
              caption="表 1-2  实验环境与超参数配置")

    H2("1.5  训练过程")
    P("先在 CPU 上对 EfficientNet-B0 做单折 30 轮训练以观察收敛行为（图 1-2）。"
      "迁移学习效果显著：第 6 轮验证准确率即达 90%；最优轮次出现在中段（第 10 轮 92%），"
      "此后多次回到 92% 但中间存在回落（第 13 轮降至 89%），故采用按验证集取最优轮次的策略。"
      "后期过拟合显现：训练损失自第 10 轮的 0.6006 降至末轮 0.5133，验证损失仅由 0.5850 "
      "缓降至 0.5545，至末轮训练损失已低于验证损失，二者差距由 0.016 扩大至 0.041，"
      "说明在 400 张样本上继续增加轮次已无收益，正则化与集成的必要性由此得到印证。")
    FIG(p12, "图 1-2  EfficientNet-B0 单折训练过程的损失与准确率曲线", 14.6)
    P("四骨干 × 五折共 20 次独立训练的验证准确率见表 1-3。ConvNeXt-Tiny 折内平均最高"
      "（94.2%），ResNet50 次之（93.0%），EfficientNet-B0 与 ResNet18 均为 92.2%；"
      "单折波动达 89%~97%（如 ResNet18 的 fold 0 为 96% 而 fold 1 仅 89%），"
      "再次说明单折评估不可靠。另需注意，前 3 轮验证准确率由 33% 快速升至 63%，"
      "正是判别式学习率让随机初始化的分类头快速到位的直接体现。")
    fold_rows = []
    for n in FOLD_ACC:
        v = FOLD_ACC[n]
        fold_rows.append([n] + ["%.2f" % x for x in v] + ["%.4f" % float(np.mean(v))])
    add_table(doc,
              ["骨干网络", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "均值"],
              fold_rows, widths=[3.3, 2.05, 2.05, 2.05, 2.05, 2.05, 2.2],
              caption="表 1-3  四骨干五折交叉验证的验证准确率")
    P("此外，各折之间存在系统性难度差异：fold 1 在四种骨干上均偏低（ResNet18 仅 89%），"
      "可能源于该折包含更多成像条件特殊的样本。这一现象提醒不应以单折成绩评判方法优劣，"
      "也正是采用五折汇总而非单折汇报的理由。")

    H2("1.6  实验结果与分析")
    H3("1.6.1  总体结果与各类别指标")
    P("四骨干 OOF 概率按骨干等权平均后，集成 OOF 准确率 92.60%、平衡准确率 92.60%、"
      "macro-F1 92.56%；三者接近说明 OOF 上未出现严重类别偏置（测试集情况见 1.6.3 节）。"
      "各类指标见表 1-4，呈清晰的两极分化。")
    rows = []
    for i, c in enumerate(CLASSES):
        rows.append([c, CLASSES_CN[i], "100", "%.3f" % OOF_RECALL[i],
                     "%.3f" % OOF_PREC[i], "%.3f" % OOF_F1CLS[i]])
    add_table(doc,
              ["类别（英）", "类别（中）", "样本数", "召回率", "精确率", "F1"],
              rows, widths=[3.6, 2.6, 2.0, 2.4, 2.4, 2.2],
              caption="表 1-4  集成模型在 OOF 上的分类指标")

    H3("1.6.2  混淆结构与集成的作用")
    P("无香母菊与甜菜的召回率均为 100%（精确率分别为 1.000、0.990），说明形态独特性强、"
      "可稳定识别；黑麦草与散穗黍召回率仅 83.0%，是全部误差的主要来源。由召回与精确率的"
      "关系可推断混淆方向：黑麦草两者同时偏低且数值接近（0.830 / 0.814），属双向混淆；"
      "散穗黍精确率（0.883）高于召回率（0.830），说明“被误判出去”多于“被误判进来”。"
      "结合普通小麦召回率高达 97.0%，可判断黑麦草与散穗黍之间的相互混淆是主要误差结构，"
      "与 1.2 节的形态学分析一致。这一结构意味着：误差并非均匀分布于五类，而是集中在"
      "黑麦草与散穗黍这一混淆对上；在 macro-F1 这类对各类等权的指标上，"
      "这两类的 F1（0.822、0.856）将主导最终得分，针对性的改进应优先投向此处。")
    P("集成的意义需按同口径评估：四骨干折内均值为 92.85%，集成 OOF 为 92.60%，略低于均值"
      "与最强的 ConvNeXt-Tiny。可见在单模型已普遍达 92%~94% 时，等权集成的主要收益不在"
      "继续抬高准确率，而在降低方差与抑制单架构的类别偏好；这也提示当成员强弱悬殊时，"
      "等权平均会被较弱成员拖累。骨干间预测一致率为 89.4%，分歧的 10.6% 正是难以判别的"
      "边界样本。此外，测试集预测置信度均值 0.812，低于 0.5 者仅 12 张（占 3.2%），"
      "表明模型对绝大多数样本把握较高，剩余少数低置信样本是最可能的错误位置。")

    H3("1.6.3  决策修正的效果与适用边界")
    P("决策权重修正的量化结果见表 1-5。修正后在 OOF 上 macro-F1 由 0.9256 升至 0.9391"
      "（+0.0135），但该数字偏乐观——权重正是在这批数据上学习的；真正可信的是留一折检验："
      "在模型完全未见过的折上，macro-F1 由 0.9254 升至 0.9348，净增益 +0.0095，"
      "方向明确而幅度有限。")
    add_table(doc,
              ["评估方式", "准确率", "平衡准确率", "macro-F1", "说明"],
              [["修正前（OOF）", "0.9260", "0.9260", "0.9256", "基准"],
               ["修正后（OOF）", "0.9400", "0.9400", "0.9391", "同批数据自评，偏乐观"],
               ["留一折检验", "—", "0.9260→0.9360", "0.9254→0.9348",
                "可信增益：F1 +0.0095"]],
              widths=[3.2, 2.2, 3.5, 3.4, 3.3],
              caption="表 1-5  决策权重修正的量化效果")
    P("然而将学到的权重应用到测试集后，出现了一个值得深究的现象（图 1-3）：黑麦草由 18.8% "
      "压至 15.1%，散穗黍由 26.7% 抬至 30.2%，平均偏离度由 5.00 恶化至 6.38。"
      "根源在于偏置来源不匹配——权重在 OOF 上学习，而集成后的 OOF 预测分布本身已接近均匀"
      "（0.204、0.206、0.188、0.200、0.202），即验证集上已几乎无偏置可修；"
      "此时优化器只能通过牺牲难分类别（学到的黑麦草权重仅 0.704，而黑麦草恰为召回最低类）"
      "来微量提升平衡准确率。这套权重施于偏置严重的测试集时方向恰好相反，遂加剧了偏斜。")
    FIG(p15, "图 1-3  决策权重修正前后的测试集预测分布", 14.2)
    P("由此引出方法二：既然测试集偏置（普通小麦 8.7%）远大于验证集，应直接在测试集上测量"
      "并修正。实验在 OOF 上验证了该方向的有效性——将 OOF 分布推向均匀后，平衡准确率由 "
      "0.9260 升至 0.9400，与决策权重修正所得一致，说明“推向均匀先验”这一操作本身有效。"
      "据此得出的方法论结论是：决策修正必须匹配偏置的来源；用验证集的偏置去修测试集的偏置，"
      "当二者幅度与方向不一致时不仅无效，反而有害——该结论由实测数据直接支撑，而非推测。")

    H3("1.6.4  提交成绩")
    P("两次提交的成绩见表 1-6。")
    add_table(doc,
              ["提交", "方法配置", "排行榜成绩"],
              [["第一次", "单骨干 + 五折交叉验证 + 8 视角 TTA", "0.9232"],
               ["第二次", "四骨干集成 + 五折 + 8 视角 TTA + 决策权重修正", "0.9312"]],
              widths=[2.6, 9.5, 3.4],
              caption="表 1-6  Kaggle 排行榜提交成绩")
    P("成绩由 0.9232 提升至 0.9312，绝对提升 0.8 个百分点。需客观说明，该提升主要来自"
      "多骨干集成与多视角 TTA 带来的方差降低；决策修正的贡献难以从总成绩中单独剥离，"
      "且 1.6.3 节的分析表明其在本次测试集上可能未产生正向作用。"
      "这一归因判断本身也是实验分析的一部分。")

    H2("1.7  创新点")
    NUM(["以“验证协议”而非“模型结构”为核心。面对每类仅 100 张的极端小样本，先建立五折分层"
         "交叉验证、按验证取最优、留一折检验的评估体系，使后续改进可被证伪；这一取舍在"
         "小样本任务中具有普遍意义。",
         "识别并量化了迁移学习的类别先验偏置。发现测试集中普通小麦被压至 8.7%（期望 20%），"
         "并将其归因于 ImageNet 标签先验的继承。该现象在常规准确率报告中被掩盖，"
         "却显著影响 macro-F1 一类宏平均指标。",
         "提出并界定了两种决策修正方法的适用边界。OOF 决策权重与直推式先验修正均不使用标签、"
         "均不涉及泄漏，但对应不同来源的偏置；实验以留一折检验与分布对比给出了"
         "“何时该用哪一种”的实证判据。",
         "增强策略的任务定制化。依据“正俯视拍摄”将旋转开放至 ±180°，依据“植株可仅占画面"
         "一角”将随机裁剪下限下探至 0.45，并基于细粒度任务特性有理据地放弃 Mixup。"])

    H2("1.8  不足与改进方向")
    P("本实验仍有以下不足：其一，伪标签半监督尚未启用——测试集 378 张与被评分集合同分布，"
      "以置信度与多骨干一致性双重闸门筛选后可将标注量扩大 70%，是小样本条件下剩余收益"
      "最大的一项；其二，直推式修正的测试集效果因无标签只能以对照提交评估，尚未量化；"
      "其三，未引入 384 高分辨率训练，可能损失叶脉级细节；其四，黑麦草与散穗黍召回率仅 83%，"
      "可考虑对易混类别单独训练二分类判别器或对易混区域做局部高分辨率裁剪；"
      "其五，集成为简单等权平均，可改为按验证表现或预测不确定性加权融合。")

    # ---------------- 参考文献 ----------------
    H2("参考文献")
    refs = [
        "[1] Deng J, Dong W, Socher R, et al. ImageNet: A Large-Scale Hierarchical Image Database[C]. CVPR, 2009.",
        "[2] Tan M, Le Q V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks[C]. ICML, 2019.",
        "[3] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016.",
        "[4] Liu Z, Mao H, Wu C Y, et al. A ConvNet for the 2020s[C]. CVPR, 2022.",
        "[5] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization[C]. ICLR, 2019.",
        "[6] Szegedy C, Liu W, Jia Y, et al. Going Deeper with Convolutions[C]. CVPR, 2015.",
        "[7] Smith L N. A Disciplined Approach to Neural Network Hyper-Parameters: Part 1[R]. arXiv:1803.09820, 2018.",
        "[8] Menon A K, Jayasumana S, Rawat A S, et al. Long-tail Learning via Logit Adjustment[C]. ICLR, 2021.",
    ]
    for r_ in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.85)
        p.paragraph_format.first_line_indent = Cm(-0.85)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        run = p.add_run(r_)
        set_run_font(run, size=8.5)

    out = os.path.join(OUT_DIR, "任务一_植物分类_实验报告.docx")
    doc.save(out)
    print("[docx] saved:", out)
    return out


if __name__ == "__main__":
    build()
