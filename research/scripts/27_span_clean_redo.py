"""Hypothesis B, redone with the spans handled correctly.

Script 07 concluded that conditioning on the gold spans does not help
evidence_quality: +0.008 with a confidence interval crossing zero. That result
now looks unsafe, because it fed the span fields to the model as single strings.

They are not single strings. Roughly a quarter of `promise_string` and
`evidence_string` values concatenate several *non-adjacent* pieces of the
paragraph with a full-width bar, and the pieces often carry a stray digit from
the source layout:

    於2021年11月正式承諾2050年達成淨零碳排放的承諾，初步規劃短、中、長期減碳策略。2
    ｜ 未來將持續追蹤減碳成果，確保達成目標。

Taken whole, only 72% of the values are verbatim substrings of the paragraph.
Split on the bar and stripped of the trailing digits, 92-98% are. So script 07
was not testing "does the evidence span help" but "does a bar-delimited
concatenation with layout noise help", which is a different and much weaker
question.

This reruns the same four conditions on cleaned spans, and keeps the dirty
version as its own condition so the size of the artefact is on the record rather
than assumed. Everything else -- model, split, bootstrap, seed -- matches script
07, so any change is attributable to the cleaning alone.

The model is still character n-gram TF-IDF, which cannot represent the
promise-evidence relation and understates what a cross-encoder would do with
condition C. That limitation is unchanged and the neural rerun is separate; the
question here is narrower and worth settling first: **was the negative result an
artefact of the preprocessing?**
"""
import csv
import io
import os
import random
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
os.makedirs(OUT, exist_ok=True)

SEED = 20260823          # same as script 07
CLASSES = ["Clear", "Not Clear", "Misleading"]
N_BOOT = 1000
SEP = "｜"           # the full-width bar used to join pieces


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def span_ok(v):
    return (v or "").strip() not in ("", "N/A")


def clean_span(raw, para):
    """Split on the bar, drop layout digits, keep pieces the paragraph contains.

    A piece that survives verbatim in the paragraph is a real annotated span. One
    that does not is usually a fragment the annotation tool clipped, and passing
    it on unchanged is what put separators and page numbers into the model's
    input in the first place.
    """
    if not span_ok(raw):
        return ""
    pieces = []
    for p in raw.split(SEP):
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"\s*\d+\s*$", "", p).strip()   # trailing layout digit
        if p:
            pieces.append(p)
    kept = [p for p in pieces if p in (para or "")]
    return "。".join(kept if kept else pieces)


def rand_span(para, n, rng):
    para = para or ""
    if len(para) <= n or n <= 0:
        return para
    i = rng.randrange(0, len(para) - n)
    return para[i:i + n]


def build(rows, rng):
    """Return the five input variants plus labels for one split."""
    out = {k: [] for k in ("A_para", "B_ev_clean", "C_pair_clean",
                           "D_random", "E_ev_dirty")}
    y = []
    for r in rows:
        para = r["data"] or ""
        ev_dirty = r["evidence_string"].strip() if span_ok(r["evidence_string"]) else ""
        ev = clean_span(r.get("evidence_string"), para)
        pr = clean_span(r.get("promise_string"), para)
        out["A_para"].append(para)
        out["B_ev_clean"].append(ev)
        out["C_pair_clean"].append((pr + " " + ev).strip())
        out["D_random"].append(rand_span(para, len(ev), rng))
        out["E_ev_dirty"].append(ev_dirty)
        y.append(r["evidence_quality"].strip())
    return out, np.array(y)


def fit_score(xtr, ytr, xte, yte):
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2,
                        sublinear_tf=True),
        LinearSVC(C=1.0, class_weight="balanced", random_state=SEED))
    pipe.fit(xtr, ytr)
    p = pipe.predict(xte)
    return f1_score(yte, p, labels=CLASSES, average="macro", zero_division=0), p


def boot_ci(yte, pa, pb, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(yte))
    d = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        ys = yte[s]
        d.append(f1_score(ys, pb[s], labels=CLASSES, average="macro", zero_division=0)
                 - f1_score(ys, pa[s], labels=CLASSES, average="macro", zero_division=0))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    rng = random.Random(SEED)
    tr = [r for r in load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
          if r["evidence_status"] == "Yes" and r["evidence_quality"].strip() in CLASSES]
    te = [r for r in load("vpesg4k_test_2000.csv")
          if r["evidence_status"] == "Yes" and r["evidence_quality"].strip() in CLASSES]

    Xtr, ytr = build(tr, rng)
    Xte, yte = build(te, rng)

    # how much did the cleaning actually change?
    n_sep = sum(1 for r in tr + te
                if span_ok(r.get("evidence_string")) and SEP in r["evidence_string"])
    dirty_sub = sum(1 for r in tr + te
                    if span_ok(r.get("evidence_string"))
                    and r["evidence_string"].strip() in (r["data"] or ""))
    clean_sub = sum(1 for i, r in enumerate(tr + te)
                    if span_ok(r.get("evidence_string"))
                    and clean_span(r["evidence_string"], r["data"] or "").replace("。", "")
                    and all(p in (r["data"] or "")
                            for p in clean_span(r["evidence_string"],
                                                r["data"] or "").split("。") if p))
    n_span = sum(1 for r in tr + te if span_ok(r.get("evidence_string")))

    L = []
    A = L.append
    A("# Hypothesis B, redone on cleaned spans")
    A("")
    A("Generated by `research/scripts/27_span_clean_redo.py`. Same model, split, "
      "bootstrap and seed as script 07; the only change is that the span fields "
      "are split on the full-width bar and stripped of trailing layout digits "
      "before use.")
    A("")
    A("## What the cleaning changes")
    A("")
    A("| | count | share |")
    A("|---|---:|---:|")
    A("| rows with an `evidence_string` | %d | 100%% |" % n_span)
    A("| ...containing the bar separator | %d | %.0f%% |" % (n_sep, 100 * n_sep / n_span))
    A("| ...verbatim substring **before** cleaning | %d | %.0f%% |"
      % (dirty_sub, 100 * dirty_sub / n_span))
    A("| ...all pieces verbatim **after** cleaning | %d | %.0f%% |"
      % (clean_sub, 100 * clean_sub / n_span))
    A("")

    A("## Conditions")
    A("")
    A("| code | input |")
    A("|---|---|")
    A("| A | full paragraph, the current formulation |")
    A("| B | cleaned `evidence_string` only |")
    A("| C | cleaned `promise_string` + cleaned `evidence_string` |")
    A("| D | control: random substring of the paragraph, same length as B |")
    A("| E | **uncleaned** `evidence_string`, i.e. what script 07 actually used |")
    A("")

    res, preds = {}, {}
    for k in ("A_para", "B_ev_clean", "C_pair_clean", "D_random", "E_ev_dirty"):
        res[k], preds[k] = fit_score(Xtr[k], ytr, Xte[k], yte)

    A("## Results")
    A("")
    A("Test macro-F1 over Clear / Not Clear / Misleading, %d training rows, "
      "%d test rows." % (len(tr), len(te)))
    A("")
    A("| condition | macro-F1 | vs A | bootstrap 95% CI |")
    A("|---|---:|---:|---|")
    for k, lab in [("A_para", "A  full paragraph"),
                   ("B_ev_clean", "B  cleaned evidence span"),
                   ("C_pair_clean", "C  cleaned promise + evidence"),
                   ("D_random", "D  random control"),
                   ("E_ev_dirty", "E  uncleaned span (script 07's input)")]:
        if k == "A_para":
            A("| %s | %.4f | --- | --- |" % (lab, res[k]))
            continue
        lo, hi = boot_ci(yte, preds["A_para"], preds[k])
        A("| %s | %.4f | %+.4f | [%+.4f, %+.4f] |"
          % (lab, res[k], res[k] - res["A_para"], lo, hi))
    A("")
    A("Script 07 reported A 0.3825, B 0.3905 (+0.0080, CI crossing zero), "
      "C 0.3915, D 0.3755.")
    A("")

    A("## Reading")
    A("")
    d_clean = res["B_ev_clean"] - res["E_ev_dirty"]
    A("Cleaning the span moves condition B by **%+.4f** (%.4f cleaned against "
      "%.4f as script 07 used it). " % (d_clean, res["B_ev_clean"], res["E_ev_dirty"]))
    if abs(d_clean) < 0.005:
        A("That is small enough that the preprocessing flaw does not explain the "
          "original negative result: hypothesis B fails for a reason other than "
          "dirty inputs.")
    else:
        A("That is large enough to matter, so the original negative result was at "
          "least partly an artefact of the preprocessing and the conclusion has "
          "to be restated.")
    A("")
    lo, hi = boot_ci(yte, preds["A_para"], preds["B_ev_clean"])
    A("Against the full paragraph, the cleaned span gives %+.4f with a 95%% "
      "interval of [%+.4f, %+.4f]. The control condition D gives %+.4f, which is "
      "what a same-length but uninformative input is worth."
      % (res["B_ev_clean"] - res["A_para"], lo, hi,
         res["D_random"] - res["A_para"]))
    A("")
    A("The model is still character n-gram TF-IDF and cannot represent the "
      "promise-evidence relation, so condition C remains understated. That "
      "limitation is unchanged from script 07 and is what the neural rerun would "
      "address.")
    A("")

    dest = os.path.join(OUT, "span_clean_redo.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
