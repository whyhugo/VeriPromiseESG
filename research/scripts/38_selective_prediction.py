"""If part of the label is not decidable from the text, the system should say so.

Everything this project has measured on `evidence_quality` points the same way.
Nine interventions failed. Four frontier models fail as badly as a quantised 8B
one and worse than a fine-tuned 330M ensemble. The rows on which those four
models agree with each other and differ from the annotators are systematically
the ones with fewer surface markers of concreteness, and the criteria the
organisers published reproduce the gold label barely above chance while the
criterion they never mention reproduces the models at 0.786.

Read together, that is not a call for a better classifier. It says a fraction of
the rows are not determined by the text a system is given, and the useful thing
to do with those rows is to decline them rather than guess.

That is only useful if the undecidable rows can be recognised *without* the
label, and script 34 supplies the reason to think they can: the features that
separate the disputed rows are computed from the paragraph and the spans, not
from the answer. So this builds a selective classifier over the existing system
and asks whether that separation converts into accuracy.

Three ways of deciding what to decline, all fitted out-of-fold and applied once
to test:

  confidence      the model's own maximum class probability, the standard
                  baseline for selective prediction
  criterion       a logistic regression over the script 34 feature families,
                  trained out-of-fold to predict whether the classifier is
                  about to be wrong
  both            confidence and the criterion features together

The question is not whether abstention helps -- it always does, since dropping
any rows raises accuracy on the rest. The question is whether the criterion
features find errors that confidence misses, which is what a criterion-mismatch
account predicts and a plain capability account does not.

CPU only. Reads condition A of script 36, which is the same backbone and the
same five-fold protocol used throughout.
"""
import argparse
import csv
import importlib.util
import io
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
S15 = os.path.join(ROOT, "research", "artifacts", "s15")
S26 = os.path.join(ROOT, "research", "artifacts", "s26")
S36 = os.path.join(ROOT, "research", "artifacts", "s36")
os.makedirs(OUT, exist_ok=True)

SEED = 20260829
EQ = ["Clear", "Not Clear", "Misleading"]
COVERAGES = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

_keep_ours = sys.stdout
_spec = importlib.util.spec_from_file_location(
    "s34", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "34_guideline_criteria_test.py"))
s34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s34)
_keep_theirs = sys.stdout


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def feats(row):
    """All three script 34 families, concatenated. Names kept for the report."""
    out = []
    for _, fn in s34.FAMILIES:
        out.extend(fn(row))
    return out


def risk_coverage(y, pred, score):
    """Accuracy and macro-F1 as the least-trusted rows are dropped.

    `score` is higher when the row should be declined, so rows are kept in
    ascending order of it.
    """
    order = np.argsort(score)
    rows = []
    for c in COVERAGES:
        k = order[:max(2, int(round(c * len(y))))]
        rows.append((c, len(k),
                     float((pred[k] == y[k]).mean()),
                     float(f1_score(y[k], pred[k], average="macro",
                                    zero_division=0)),
                     float(balanced_accuracy_score(y[k], pred[k]))))
    return rows


def auarc(y, pred, score, grid=None):
    """Area under the accuracy-coverage curve; 1.0 is a perfect ranker."""
    order = np.argsort(score)
    grid = grid if grid is not None else np.linspace(0.05, 1.0, 40)
    acc = []
    for c in grid:
        k = order[:max(2, int(round(c * len(y))))]
        acc.append((pred[k] == y[k]).mean())
    trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed it
    return float(trapz(acc, grid) / (grid[-1] - grid[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    rng = np.random.default_rng(SEED)

    trv = [r for r in load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
           if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ]
    te = [r for r in load("vpesg4k_test_2000.csv")
          if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ]
    ytr = np.array([EQ.index(na(r["evidence_quality"])) for r in trv])
    yte = np.array([EQ.index(na(r["evidence_quality"])) for r in te])

    z = np.load(os.path.join(S36, "cond_A.npz"))
    p_oof, p_te = z["oof"], z["test"]
    pred_oof, pred_te = p_oof.argmax(1), p_te.argmax(1)
    conf_oof, conf_te = p_oof.max(1), p_te.max(1)
    err_oof = (pred_oof != ytr).astype(int)
    err_te = (pred_te != yte).astype(int)

    names = [n for n, _ in feats(trv[0])]
    Xtr = np.array([[v for _, v in feats(r)] for r in trv], float)
    Xte = np.array([[v for _, v in feats(r)] for r in te], float)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    # Every abstention rule is fitted on out-of-fold rows only.
    SIGNALS = {}
    SIGNALS["confidence"] = (-conf_oof, -conf_te)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, err_oof)
    SIGNALS["criterion"] = (m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1])
    Atr = np.column_stack([Xtr, -conf_oof])
    Ate = np.column_stack([Xte, -conf_te])
    m2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Atr, err_oof)
    SIGNALS["both"] = (m2.predict_proba(Atr)[:, 1], m2.predict_proba(Ate)[:, 1])
    SIGNALS["random"] = (rng.random(len(ytr)), rng.random(len(yte)))

    L = []
    A = L.append
    A("# Declining the rows the text does not decide")
    A("")
    A("Generated by `research/scripts/38_selective_prediction.py`. The base "
      "classifier is condition A of script 36 -- `%s` over the full paragraph, "
      "five folds, the same protocol as the rest of the project. Selective "
      "prediction leaves that classifier untouched and only chooses which of "
      "its predictions to keep."
      % "hfl/chinese-roberta-wwm-ext-large")
    A("")
    A("Base accuracy is %.4f on the %d test rows, macro-F1 %.4f. Every "
      "abstention rule below is fitted on the %d out-of-fold train+val rows and "
      "applied once to test."
      % ((pred_te == yte).mean(), len(yte),
         f1_score(yte, pred_te, average="macro", zero_division=0), len(ytr)))
    A("")

    A("## Can an error be predicted before it is made?")
    A("")
    A("Ranking the test rows by each signal and asking how well it orders the "
      "classifier's errors. 0.500 is chance.")
    A("")
    A("| signal | AUC for predicting an error | area under accuracy-coverage |")
    A("|---|---:|---:|")
    for k, (_, s_te) in SIGNALS.items():
        A("| %s | %.4f | %.4f |"
          % (k, roc_auc_score(err_te, s_te), auarc(yte, pred_te, s_te)))
    A("")

    A("## Risk-coverage")
    A("")
    A("Accuracy and macro-F1 on the rows the system keeps, as it declines more.")
    A("")
    A("| coverage | kept | " + " | ".join(
        "%s acc." % k for k in SIGNALS) + " |")
    A("|---" * (2 + len(SIGNALS)) + "|")
    curves = {k: risk_coverage(yte, pred_te, s) for k, (_, s) in SIGNALS.items()}
    for i, c in enumerate(COVERAGES):
        A("| %.0f%% | %d | %s |"
          % (100 * c, curves["confidence"][i][1],
             " | ".join("%.4f" % curves[k][i][2] for k in SIGNALS)))
    A("")
    A("| coverage | " + " | ".join("%s macro-F1" % k for k in SIGNALS) + " |")
    A("|---" * (1 + len(SIGNALS)) + "|")
    for i, c in enumerate(COVERAGES):
        A("| %.0f%% | %s |"
          % (100 * c, " | ".join("%.4f" % curves[k][i][3] for k in SIGNALS)))
    A("")

    A("## What the criterion signal is using")
    A("")
    A("Coefficients of the out-of-fold error model, largest first. A positive "
      "weight means the feature makes an error more likely.")
    A("")
    A("| feature | weight |")
    A("|---|---:|")
    for j in np.argsort(-np.abs(m.coef_[0]))[:10]:
        A("| %s | %+.3f |" % (names[j], m.coef_[0][j]))
    A("")

    # --- does it decline the rows the frontier models also disputed? ---------
    if os.path.isdir(S26):
        fm = {}
        for f in sorted(os.listdir(S26)):
            if f.endswith(".json"):
                got = json.load(open(os.path.join(S26, f), encoding="utf-8"))
                fm[f.split("__")[0]] = {k: (v[0] or "Clear") for k, v in got.items()}
        models = sorted(fm)
        idx = {r["id"]: n for n, r in enumerate(te)}
        unan = [i for i in idx if all(i in fm[m_] for m_ in models)
                and len({fm[m_][i] for m_ in models}) == 1]
        disputed = np.zeros(len(te), bool)
        for i in unan:
            if fm[models[0]][i] != na(te[idx[i]]["evidence_quality"]):
                disputed[idx[i]] = True
        A("## Are the declined rows the disputed ones?")
        A("")
        A("The %d rows on which all four frontier models agreed with each other "
          "and differed from the gold label were identified in script 33 "
          "without reference to this classifier. If the criterion signal is "
          "measuring what it claims to, it should decline them first."
          % int(disputed.sum()))
        A("")
        A("| signal | share of disputed rows declined at 80%% coverage | "
          "share of all rows |")
        A("|---|---:|---:|")
        for k, (_, s_te) in SIGNALS.items():
            order = np.argsort(s_te)
            dropped = np.zeros(len(te), bool)
            dropped[order[int(round(0.8 * len(te))):]] = True
            A("| %s | %.3f | %.3f |"
              % (k, dropped[disputed].mean(), dropped.mean()))
        A("")

    A("## Reading")
    A("")
    A("Abstention always raises accuracy on what is kept, so the comparison "
      "that means anything is between signals at the same coverage, and "
      "against the random column, which is what pure coverage reduction buys.")
    A("")
    A("A criterion-mismatch account predicts that the guideline-independent "
      "features find errors the model's own confidence does not, because those "
      "errors are not failures of capability: the model is confident and "
      "consistent, it is applying a criterion the annotators did not use. A "
      "plain capability account predicts confidence should dominate.")
    A("")
    dest = os.path.join(OUT, "selective_prediction.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
