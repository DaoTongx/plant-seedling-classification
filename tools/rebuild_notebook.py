# -*- coding: utf-8 -*-
"""
Re-embed src/plant_classifier.py into the notebook's PIPELINE_SOURCE cell.

The notebook carries the whole pipeline as one Python string so that a single
.ipynb file is self-contained on Kaggle.  Editing that string by hand is
error-prone, so this script does it deterministically:

    python tools/rebuild_notebook.py

It rewrites the PIPELINE_SOURCE cell from the current src/plant_classifier.py,
leaves every other cell untouched, and verifies the result.

Use tools/build_notebook.py instead if you want the whole notebook regenerated
from scratch (it also rewrites the prose cells).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SRC = os.path.join(PROJECT, "src", "plant_classifier.py")
NB = os.path.join(PROJECT, "notebooks", "plant_classification_kaggle.ipynb")


def main():
    if not os.path.exists(SRC):
        print("missing %s" % SRC)
        return 1
    if not os.path.exists(NB):
        print("missing %s" % NB)
        return 1

    with open(SRC, encoding="utf-8") as fh:
        source = fh.read()

    # The cell assigns the pipeline to a variable using an r''' ... ''' literal.
    # Guard against the source ever containing that delimiter.
    if "'''" in source:
        print("ERROR: source contains a triple single quote, which would break "
              "the embedded string literal. Replace it with \"\"\" in "
              "src/plant_classifier.py and retry.")
        return 1

    new_cell_source = "PIPELINE_SOURCE = r'''\n" + source + "\n'''"

    with open(NB, encoding="utf-8") as fh:
        nb = json.load(fh)

    hits = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell["source"])
        if text.startswith("PIPELINE_SOURCE"):
            cell["source"] = new_cell_source.splitlines(True)
            hits += 1

    if hits != 1:
        print("ERROR: expected exactly one PIPELINE_SOURCE cell, found %d" % hits)
        return 1

    with open(NB, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    # ---- verify ----
    with open(NB, encoding="utf-8") as fh:
        check = json.load(fh)
    embedded = None
    for cell in check["cells"]:
        if cell.get("cell_type") == "code":
            text = "".join(cell["source"])
            if text.startswith("PIPELINE_SOURCE"):
                embedded = text
    ok = embedded is not None and source in embedded
    print("re-embedded %d characters of pipeline source" % len(source))
    print("embedded text matches src/plant_classifier.py: %s" % ok)
    if not ok:
        return 1

    # report the notable new symbols so the user can confirm the right version
    for name in ("tune_decision_weights", "fit_and_validate_weights",
                 "pseudo_label_train", "select_pseudo_labels", "macro_f1"):
        print("  contains %-24s %s" % (name, name in embedded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
