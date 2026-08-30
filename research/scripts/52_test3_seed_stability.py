# -*- coding: utf-8 -*-
"""Is condition D's interval a property of the data or of the RNG seed?

Script 36 reports A-D as [-0.0389, -0.0003]: an upper bound 0.0003 from zero, on
one draw of a bootstrap that also does not pin labels. Re-run the same
computation across many seeds and count how often it excludes zero.
"""
import csv
import io
import os

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
S36 = os.path.join(ROOT, "research", "artifacts", "s36")
OUT = os.path.join(ROOT, "research", "output", "test3_seed_stability.md")
EQ = ["Clear", "Not Clear", "Misleading"]


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


te = [r for r in csv.DictReader(open(os.path.join(D, "vpesg4k_test_2000.csv"),
                                     encoding="utf-8-sig"))
      if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ]
y = np.array([EQ.index(na(r["evidence_quality"])) for r in te])

P = {}
for c in "ABCDEF":
    f = os.path.join(S36, "cond_%s.npz" % c)
    if os.path.exists(f):
        P[c] = np.load(f)["test"].argmax(1)

L = ["rows=%d  conditions=%s" % (len(y), "".join(sorted(P)))]
L.append("predicted Misleading counts: %s"
         % {c: int((P[c] == 2).sum()) for c in sorted(P)})
L.append("")
L.append("%-6s %10s %12s %12s %8s" % ("cond", "point", "mean lo", "mean hi",
                                      "excl 0 / 40"))
for c in sorted(P):
    if c == "A":
        continue
    point = (f1_score(y, P[c], average="macro", zero_division=0)
             - f1_score(y, P["A"], average="macro", zero_division=0))
    los, his, excl = [], [], 0
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        d = []
        for _ in range(4000):
            k = rng.integers(0, len(y), len(y))
            if len(set(y[k])) < 2:
                continue
            d.append(f1_score(y[k], P[c][k], average="macro", zero_division=0)
                     - f1_score(y[k], P["A"][k], average="macro", zero_division=0))
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        los.append(lo)
        his.append(hi)
        excl += int(lo > 0 or hi < 0)
    L.append("%-6s %10.4f %12.4f %12.4f %8d" % (c, point, np.mean(los),
                                                np.mean(his), excl))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", OUT)
