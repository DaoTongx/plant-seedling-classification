# -*- coding: utf-8 -*-
"""
把代码仓库地址写入报告生成脚本的 GITHUB_URL 常量，并重新生成 Word 报告。

用法：
    python set_github_url.py https://github.com/<用户名>/<仓库名>
    python set_github_url.py --clear          # 清空地址，报告中不生成该小节

注意：脚本只修改 build_report_compact.py 中的一行常量，不做其他改动。
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "build_report_compact.py")
PATTERN = re.compile(r'^GITHUB_URL\s*=\s*".*?"\s*$', re.MULTILINE)


def main(argv):
    if not argv:
        print(__doc__)
        return 2

    url = "" if argv[0] == "--clear" else argv[0].strip()
    if url and not re.match(r"^https?://", url):
        print("地址需以 http:// 或 https:// 开头：%r" % url)
        return 2

    src = open(TARGET, encoding="utf-8").read()
    if not PATTERN.search(src):
        print("未在 %s 中找到 GITHUB_URL 常量" % TARGET)
        return 1

    new = PATTERN.sub('GITHUB_URL = "%s"' % url, src, count=1)
    open(TARGET, "w", encoding="utf-8", newline="\n").write(new)
    print("GITHUB_URL -> %r" % url)

    r = subprocess.run([sys.executable, TARGET], cwd=HERE)
    if r.returncode != 0:
        print("报告生成失败")
        return r.returncode

    # 确认小节确实写进了文档
    import docx
    d = docx.Document(os.path.join(HERE, "任务一_植物分类_实验报告.docx"))
    has = any("代码与复现" in p.text for p in d.paragraphs)
    print("报告中含「1.9 代码与复现」小节：%s" % ("是" if has else "否"))
    if url and not has:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
