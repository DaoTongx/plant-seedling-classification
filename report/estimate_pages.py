# -*- coding: utf-8 -*-
"""
按版面几何估算页数。

Word COM 在本机自动化会挂起，因此改用几何模型：统计正文行数、标题高度、
表格行高、图片实际高度（从 inline_shapes 读取），累加后除以每页可用高度。
"""
import glob
import re
import sys

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

USABLE_W = 15.8                      # cm
USABLE_H = 29.7 - 2.3 - 2.1 - 1.0    # 页高 - 上下边距 - 页脚占用
CHAR_W = 0.375                       # 10.5pt 汉字平均字宽
LINE_PITCH = 0.67                    # 10.5pt、1.5 倍行距的单行步距
TABLE_ROW_H = 0.44                   # 8.8pt 单行表格行高
BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"


def estimate(path):
    d = docx.Document(path)
    img_total = sum(s.height.cm for s in d.inline_shapes)

    chars = lines = 0
    heading = table = caption = 0.0

    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            if child.findall(".//" + BLIP):
                continue                      # 图片高度已单独统计
            p = Paragraph(child, d)
            t = p.text.strip()
            if not t:
                continue
            st = p.style.name
            if st.startswith("Heading"):
                heading += 0.62 if st == "Heading 1" else 0.50
            elif re.match(r"^[图表] \d-\d", t):
                caption += 0.48
            elif t.startswith("["):
                lines += 1                    # 参考文献：8.5pt 单倍行距
            else:
                chars += len(t)
                lines += max(1, -(-len(t) // int(USABLE_W / CHAR_W)))
        elif tag == "tbl":
            table += len(Table(child, d).rows) * TABLE_ROW_H

    text_cm = lines * LINE_PITCH
    total = text_cm + heading + table + caption + img_total + 2.0
    return dict(path=path, chars=chars, lines=lines, text=text_cm,
                heading=heading, table=table, caption=caption,
                img=img_total, total=total, pages=total / USABLE_H)


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(glob.glob("*.docx"))
    for f in targets:
        r = estimate(f)
        print("文件: %s" % r["path"])
        print("  正文字数 %d，正文 %d 行 (%.1f cm)" % (r["chars"], r["lines"], r["text"]))
        print("  标题 %.1f + 表格 %.1f + 题注 %.1f + 图片 %.1f cm" %
              (r["heading"], r["table"], r["caption"], r["img"]))
        print("  内容总高 %.1f cm  ->  约 %.1f 页（每页 %.1f cm）" %
              (r["total"], r["pages"], USABLE_H))
        print()
