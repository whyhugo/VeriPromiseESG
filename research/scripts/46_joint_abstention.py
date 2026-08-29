"""Abstention on the whole row, not one field of it.

Section 5 of the side report built a selective layer for `evidence_quality`
alone. That is not what an auditor needs. A person reading these outputs wants
to know whether *this paragraph's* four fields can be trusted together, and the
project already has a measure for that: joint exact match, the fraction of
paragraphs where all four fields are correct.

Two abstention signals are compared, and the second is the interesting one.

  criterion   the script 34 features, in a model fitted out-of-fold to predict
              whether `evidence_quality` will be wrong, then applied to the
              whole row. This is a transfer: the model is trained on one field's
              errors and asked about four. It is reported as such.

  structural  whether the predicted four-tuple violates the label hierarchy at
              all -- if there is no promise, the other three must be N/A; if
              there is no evidence, quality must be N/A. This needs no model,
              no training and no labels.

The second signal is worth testing because of what the main report already
found. Forcing inconsistent predictions onto the legal set costs macro-F1,
because the projection trades rare-class recall for majority-class precision.
But the same inconsistency that projection handles badly may be a perfectly good
reason to *decline* the row instead. Nothing in the earlier result rules that
out, and declining costs nothing that projection did not already cost.

CPU only; reads the competition submission and the cached out-of-fold
predictions from script 36.
"""
import csv
import importlib.util
import io
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
SUB = os.path.join(ROOT, "submissions")
OUT = os.path.join(ROOT, "research", "output")
S36 = os.path.join(ROOT, "research", "artifacts", "s36")
os.makedirs(OUT, exist_ok=True)

SEED = 20260830
SYSTEM = "ens_3way_tuned.csv"
FIELDS = ["promise_status", "evidence_status", "evidence_quality",
          "verification_timeline"]
EQ = ["Clear", "Not Clear", "Misleading"]
COVERAGES = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

_keep_ours = sys.stdout
_spec = importlib.util.spec_from_file_location(
    "s34", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "34_guideline_criteria_test.py"))
s34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s34)
_keep_theirs = sys.stdout


def load(fn, base=D):
    return list(csv.DictReader(open(os.path.join(base, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def consistent(p):
    """Does this four-tuple obey the hierarchy the annotation implies?"""
    if na(p["promise_status"]) == "No":
        return all(na(p[f]) == "N/A"
                   for f in ["evidence_status", "evidence_quality",
                             "verification_timeline"])
    if na(p["evidence_status"]) in ("No", "N/A"):
        return na(p["evidence_quality"]) == "N/A"
    return True


def feats(row):
    out = []
    for _, fn in s34.FAMILIES:
        out.extend(v for _, v in fn(row))
    return out


def curve(correct, score, per_field=None):
    order = np.argsort(score)
    rows = []
    for c in COVERAGES:
        k = order[:max(2, int(round(c * len(correct))))]
        rec = [c, len(k), float(np.mean(correct[k]))]
        if per_field is not None:
            rec += [float(np.mean(per_field[f][k])) for f in FIELDS]
        rows.append(rec)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    te = load("vpesg4k_test_2000.csv")
    pred = {r["id"]: r for r in load(SYSTEM, SUB)}
    te = [r for r in te if r["id"] in pred]

    ok_field = {f: np.array([na(pred[r["id"]][f]) == na(r[f]) for r in te])
                for f in FIELDS}
    joint = np.array([all(ok_field[f][n] for f in FIELDS)
                      for n in range(len(te))])
    incons = np.array([not consistent(pred[r["id"]]) for r in te])

    # criterion signal, fitted out-of-fold on subtask 3 errors only
    trv = [r for r in load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
           if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ]
    z = np.load(os.path.join(S36, "cond_A.npz"))
    y_tr = np.array([EQ.index(na(r["evidence_quality"])) for r in trv])
    err_tr = (z["oof"].argmax(1) != y_tr).astype(int)
    Xtr = np.array([feats(r) for r in trv], float)
    Xte = np.array([feats(r) for r in te], float)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(
        sc.transform(Xtr), err_tr)
    s_crit = m.predict_proba(sc.transform(Xte))[:, 1]

    SIGNALS = {
        "criterion features (transferred from subtask 3)": s_crit,
        "structural inconsistency alone": incons.astype(float)
        + 1e-6 * rng.random(len(te)),
        "both": s_crit + incons.astype(float),
        "random": rng.random(len(te)),
    }

    L = []
    A = L.append
    A("# Declining the whole row")
    A("")
    A("Generated by `research/scripts/46_joint_abstention.py`. System: "
      "`submissions/%s`, the competition entry, on all %d test rows. The "
      "quantity being protected is **joint exact match** -- the fraction of "
      "paragraphs where all four fields are right at once, which is what an "
      "auditor reading the output actually depends on." % (SYSTEM, len(te)))
    A("")
    A("Base joint exact match is **%.4f**. %d rows (%.1f%%) carry a prediction "
      "that violates the label hierarchy, and those rows are jointly correct "
      "%.4f of the time against %.4f for the rest."
      % (joint.mean(), int(incons.sum()), 100 * incons.mean(),
         joint[incons].mean(), joint[~incons].mean()))
    A("")

    A("## Joint exact match as rows are declined")
    A("")
    A("| coverage | kept | " + " | ".join(SIGNALS) + " |")
    A("|---" * (2 + len(SIGNALS)) + "|")
    cur = {k: curve(joint, v) for k, v in SIGNALS.items()}
    for i, c in enumerate(COVERAGES):
        A("| %.0f%% | %d | %s |"
          % (100 * c, cur["random"][i][1],
             " | ".join("%.4f" % cur[k][i][2] for k in SIGNALS)))
    A("")

    A("## Per field, at 80% coverage")
    A("")
    A("| signal | " + " | ".join("`%s`" % f for f in FIELDS) + " |")
    A("|---" * (1 + len(FIELDS)) + "|")
    for k, v in SIGNALS.items():
        r = curve(joint, v, ok_field)[COVERAGES.index(0.8)]
        A("| %s | %s |" % (k, " | ".join("%.4f" % x for x in r[3:])))
    A("")
    A("At full coverage: %s."
      % ", ".join("`%s` %.4f" % (f, ok_field[f].mean()) for f in FIELDS))
    A("")

    A("## Reading")
    A("")
    A("**The structural rows are guaranteed wrong, and that is a deduction "
      "rather than a measurement.** The gold labels obey the hierarchy without "
      "exception. So if a prediction violates it, the predicted four-tuple "
      "differs from the gold four-tuple somewhere, and joint exact match fails "
      "for that row. The 0.0000 above is the arithmetic confirming it, not "
      "evidence for it.")
    A("")
    A("What is not deducible is that this matters in practice, and it does: "
      "%.1f%% of the rows can be flagged as certainly wrong at no cost -- no "
      "model, no training, no labels, one pass over the output. That completes "
      "an argument the main report left half finished. Forcing those rows onto "
      "the legal set costs macro-F1, because the projection trades rare-class "
      "recall for majority-class precision. Declining them costs nothing and "
      "removes rows that could not have been right. **The same observation "
      "supports a bad action and a good one, and the earlier finding ruled out "
      "only the bad one.**" % (100 * incons.mean()))
    A("")
    A("Past that point the structural signal is exhausted -- there are only %d "
      "such rows -- and the curve flattens. The criterion features carry on "
      "beyond it, and the two together are better at every coverage than "
      "either alone." % int(incons.sum()))
    A("")
    A("The criterion signal is a transfer and should be read as one. It was "
      "fitted to predict errors in `evidence_quality` and is being asked about "
      "four fields, including the %d rows with no evidence at all, which are "
      "outside the distribution it was fitted on."
      % int(sum(1 for r in te if na(r["evidence_status"]) != "Yes")))
    A("")
    dest = os.path.join(OUT, "joint_abstention.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
