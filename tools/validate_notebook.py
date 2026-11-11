import json, ast, sys
nb = json.load(open(r"notebooks\plant_classification_kaggle.ipynb", encoding="utf-8"))
print("cells:", len(nb["cells"]), "| nbformat", nb["nbformat"], ".", nb["nbformat_minor"])
bad = 0
for i, c in enumerate(nb["cells"]):
    assert c["cell_type"] in ("code","markdown"), c["cell_type"]
    assert isinstance(c["source"], list)
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad += 1
            print("SYNTAX ERROR in cell %d: %s" % (i, e))
    if c["cell_type"]=="code" and i==5:
        print("pipeline-embed cell length: %d chars" % len("".join(c["source"])))
print("syntax-valid code cells:", sum(1 for c in nb["cells"] if c["cell_type"]=="code") - bad)
# confirm the embedded pipeline matches the on-disk source byte-for-byte
src_file = open(r"src\plant_classifier.py", encoding="utf-8").read()
embed = None
for c in nb["cells"]:
    if c["cell_type"]=="code" and "".join(c["source"]).startswith("PIPELINE_SOURCE"):
        embed = "".join(c["source"])
        break
ok = embed is not None and src_file in embed
print("embedded pipeline identical to src/plant_classifier.py:", ok)
