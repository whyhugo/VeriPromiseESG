# -*- coding: utf-8 -*-
"""F3：爭議組 vs 對照組的差異，是不是只是「模型說 Not Clear vs 說 Clear」？

寫進檔案而非 stdout，避開 script 35 匯入時重綁 sys.stdout 的問題。
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = r"D:\programing\swiftx\VeriPromiseESG"
_keep = sys.stdout
sp = importlib.util.spec_from_file_location(
    "s35", os.path.join(ROOT, "research", "scripts", "35_explanation_probe.py"))
s35 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(s35)
_keep2 = sys.stdout

S26 = os.path.join(ROOT, "research", "artifacts", "s26")
S35 = os.path.join(ROOT, "research", "artifacts", "s35")
EQ = ["Clear", "Not Clear", "Misleading"]
FIG = "unstated: no concrete figures"
VAG = "stated: vague wording"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f3.txt")
L = []

te = s35.load("vpesg4k_test_2000.csv")
rows = {r["id"]: r for r in te if s35.na(r["evidence_status"]) == "Yes"
        and s35.na(r["evidence_quality"]) in EQ}
fr = {}
for f in sorted(os.listdir(S26)):
    if f.endswith(".json"):
        g = json.load(open(os.path.join(S26, f), encoding="utf-8"))
        fr[f.split("__")[0]] = {k: (v[0] or "Clear") for k, v in g.items()}
nm = sorted(fr)
ids = [i for i in rows if all(i in fr[m] for m in nm)]
un = [i for i in ids if len({fr[m][i] for m in nm}) == 1]
wrong = [i for i in un if fr[nm[0]][i] != s35.na(rows[i]["evidence_quality"])]
right = [i for i in un if fr[nm[0]][i] == s35.na(rows[i]["evidence_quality"])]
rng = np.random.default_rng(20260829)
ctrl = [right[k] for k in rng.choice(len(right), size=200, replace=False)]

got = {}
for m in nm:
    p = os.path.join(S35, "%s.json" % m)
    if os.path.exists(p):
        got[m] = json.load(open(p, encoding="utf-8"))

L.append("兩組裡，四模型共同給的答案是什麼")
for gname, grp in (("對照組", ctrl), ("爭議組", wrong)):
    c = sum(1 for i in grp if fr[nm[0]][i] == "Clear")
    n = sum(1 for i in grp if fr[nm[0]][i] == "Not Clear")
    L.append("  %-5s n=%3d   模型說 Clear %3d   說 Not Clear %3d   其他 %d"
             % (gname, len(grp), c, n, len(grp) - c - n))

L.append("")
L.append("把「模型自己的答案」固定住之後，兩組還有差別嗎？")
L.append("%-30s %7s %9s %9s" % ("分層", "理由數", "缺數字", "措辭模糊"))
for gname, grp in (("對照組", ctrl), ("爭議組", wrong)):
    for verdict in ["Clear", "Not Clear"]:
        sub = [i for i in grp if fr[nm[0]][i] == verdict]
        rs = [got[m][i][1] for m in got for i in sub
              if i in got[m] and got[m][i][1]]
        if not rs:
            continue
        L.append("%-30s %7d %9.3f %9.3f"
                 % ("%s / 模型說 %s" % (gname, verdict), len(rs),
                    np.mean([FIG in s35.code(r) for r in rs]),
                    np.mean([VAG in s35.code(r) for r in rs])))

L.append("")
L.append("共現率（說模糊時同句也說缺數字）——這一項受影響嗎？")
for gname, grp in (("對照組", ctrl), ("爭議組", wrong)):
    for verdict in [None, "Clear", "Not Clear"]:
        sub = grp if verdict is None else [i for i in grp if fr[nm[0]][i] == verdict]
        rs = [got[m][i][1] for m in got for i in sub
              if i in got[m] and got[m][i][1]]
        v = [r for r in rs if VAG in s35.code(r)]
        if len(v) < 10:
            continue
        L.append("  %-6s %-12s 說模糊 n=%4d   其中也說缺數字 %.3f"
                 % (gname, verdict or "（全部）", len(v),
                    np.mean([FIG in s35.code(r) for r in v])))

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
