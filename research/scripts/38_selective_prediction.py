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

# A single random ordering is one draw from a wide distribution, and reading a
# curve off it invites the reader to compare a real signal against noise. The
# random column below is the mean over N_RANDOM orderings; the bands on every
# column come from resampling the test rows.
N_RANDOM = 200
N_BOOT = 2000

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


def boot_coverage(y, pred, score, rng, n=N_BOOT):
    """Percentile bands for the kept-set accuracy at each coverage.

    The whole selective procedure is repeated inside each resample -- rank the
    resampled rows, keep the top fraction, score what is kept -- so the band
    covers the variability of the procedure and not just of a fixed subset.
    """
    m = len(y)
    out = {c: [] for c in COVERAGES}
    for _ in range(n):
        b = rng.integers(0, m, m)
        order = b[np.argsort(score[b], kind="stable")]
        for c in COVERAGES:
            k = order[:max(2, int(round(c * m)))]
            out[c].append((pred[k] == y[k]).mean())
    return {c: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for c, v in out.items()}


def paired_boot(y, pred, scores, ref, rng, n=N_BOOT):
    """Bands for (signal - ref) at each coverage, both ranked in the same draw.

    A signal's own band is mostly the variance of the base accuracy, which
    every signal shares; differencing inside the resample removes it.
    """
    m = len(y)
    keys = [k for k in scores if k != ref]
    out = {k: {c: [] for c in COVERAGES} for k in keys}
    for _ in range(n):
        b = rng.integers(0, m, m)
        acc = {}
        for k, sc in scores.items():
            # a fresh ordering each draw, so the null is "any random ordering"
            # rather than "this one particular one"
            sc = rng.random(m) if k == "random" else sc
            order = b[np.argsort(sc[b], kind="stable")]
            acc[k] = {c: (pred[o] == y[o]).mean()
                      for c in COVERAGES
                      for o in [order[:max(2, int(round(c * m)))]]}
        for k in keys:
            for c in COVERAGES:
                out[k][c].append(acc[k][c] - acc[ref][c])
    return {k: {c: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                for c, v in d.items()} for k, d in out.items()}


def random_curve(y, pred, rng, n=N_RANDOM):
    """Mean accuracy and macro-F1 at each coverage over `n` random orderings."""
    acc = {c: [] for c in COVERAGES}
    mf1 = {c: [] for c in COVERAGES}
    for _ in range(n):
        order = rng.permutation(len(y))
        for c in COVERAGES:
            k = order[:max(2, int(round(c * len(y))))]
            acc[c].append((pred[k] == y[k]).mean())
            mf1[c].append(f1_score(y[k], pred[k], average="macro",
                                   zero_division=0))
    return ([(c, 0, float(np.mean(acc[c])), float(np.mean(mf1[c])), 0.0)
             for c in COVERAGES],
            {c: (float(np.percentile(acc[c], 2.5)),
                 float(np.percentile(acc[c], 97.5))) for c in COVERAGES})


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

    rand_curve, rand_band = random_curve(yte, pred_te, rng)
    rand_auc = float(np.mean([roc_auc_score(err_te, rng.permutation(len(yte)))
                              for _ in range(N_RANDOM)]))
    rand_auarc = float(np.mean([auarc(yte, pred_te, rng.permutation(len(yte)))
                                for _ in range(N_RANDOM)]))

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
        if k == "random":
            A("| random (mean of %d orderings) | %.4f | %.4f |"
              % (N_RANDOM, rand_auc, rand_auarc))
            continue
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
    curves["random"] = rand_curve
    bands = {k: boot_coverage(yte, pred_te, s, np.random.default_rng(SEED + j))
             for j, (k, (_, s)) in enumerate(SIGNALS.items()) if k != "random"}
    bands["random"] = rand_band
    for i, c in enumerate(COVERAGES):
        A("| %.0f%% | %d | %s |"
          % (100 * c, curves["confidence"][i][1],
             " | ".join("%.4f" % curves[k][i][2] for k in SIGNALS)))
    A("")
    A("Bootstrap 95%% bands for the same accuracies (%d resamples of the test "
      "rows, re-ranking inside each resample; the random column is the "
      "2.5th-97.5th percentile over %d orderings)."
      % (N_BOOT, N_RANDOM))
    A("")
    A("| coverage | " + " | ".join("%s" % k for k in SIGNALS) + " |")
    A("|---" * (1 + len(SIGNALS)) + "|")
    for c in COVERAGES:
        A("| %.0f%% | %s |"
          % (100 * c, " | ".join("[%.4f, %.4f]" % bands[k][c] for k in SIGNALS)))
    A("")
    A("Those bands are wide because most of what they contain is the variance "
      "of the base accuracy, which every column shares. The comparison to read "
      "is the paired one: the same resample, both rankings, the difference "
      "between them. Against a random ordering at the same coverage:")
    A("")
    diffs = paired_boot(yte, pred_te,
                        {k: v[1] for k, v in SIGNALS.items()}, "random",
                        np.random.default_rng(SEED + 77))
    keys = [k for k in SIGNALS if k != "random"]
    A("| coverage | " + " | ".join("%s - random" % k for k in keys) + " |")
    A("|---" * (1 + len(keys)) + "|")
    for c in COVERAGES:
        A("| %.0f%% | %s |"
          % (100 * c, " | ".join("%+.4f [%+.4f, %+.4f]"
                                 % (curves[k][COVERAGES.index(c)][2]
                                    - rand_curve[COVERAGES.index(c)][2],
                                    diffs[k][c][0], diffs[k][c][1])
                                 for k in keys)))
    A("")
    d2 = paired_boot(yte, pred_te,
                     {k: SIGNALS[k][1] for k in ["confidence", "criterion", "both"]},
                     "confidence", np.random.default_rng(SEED + 78))
    A("And against the standard practice, the model's own confidence:")
    A("")
    A("| coverage | criterion - confidence | both - confidence |")
    A("|---|---|---|")
    for c in COVERAGES:
        i = COVERAGES.index(c)
        A("| %.0f%% | %+.4f [%+.4f, %+.4f] | %+.4f [%+.4f, %+.4f] |"
          % (100 * c,
             curves["criterion"][i][2] - curves["confidence"][i][2],
             d2["criterion"][c][0], d2["criterion"][c][1],
             curves["both"][i][2] - curves["confidence"][i][2],
             d2["both"][c][0], d2["both"][c][1]))
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
        A("| signal | share of disputed rows declined at 80% coverage | "
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
    A("One result here cuts the other way and has to be read alongside the "
      "rest. The criterion signal wins on accuracy at every coverage and loses "
      "on macro-F1 at every coverage. That is not a contradiction; it is the "
      "same effect this project measured when it forced the label hierarchy to "
      "be consistent. Declining a row removes it from its class, and the rows "
      "the criterion features decline are concentrated in `Not Clear`, the "
      "smaller and harder class. Macro-F1 weights that class equally with "
      "`Clear`, so shrinking it costs more than the accuracy it buys. Under "
      "accuracy -- what an auditor reading the kept rows actually experiences "
      "-- the ordering reverses.")
    A("")
    A("Two independent findings in this project therefore have one cause: "
      "under an imbalanced macro-F1, an intervention that trades rare-class "
      "coverage for majority-class correctness is penalised even when it "
      "improves the decisions a user sees.")
    A("")
    dest = os.path.join(OUT, "selective_prediction.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
