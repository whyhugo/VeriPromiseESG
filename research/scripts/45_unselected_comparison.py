"""A fair comparison, and whether a differently-biased model is worth adding.

Script 43 reported that Qwen3-8B is right on 40% of the rows where four frontier
models are all wrong. That comparison is not fair and the reason is worth being
explicit about: those 149 rows were selected *because* the four frontier models
failed on them. Any fifth model gets a free advantage there, and the size of the
advantage says nothing about the fifth model. Two of the three open models tried
made the problem obvious by scoring well on that set while answering `Clear` to
every row put in front of them.

This is the comparison that was needed instead: every system on all %d test rows
with evidence, none of which were selected using any model's predictions. Three
measurements, because accuracy alone hides a model that has stopped
discriminating:

  macro-F1              the competition's own measure, kept for continuity
  balanced accuracy     insensitive to the 83/17 class split
  discrimination gap    P(say Clear | gold Clear) - P(say Clear | gold Not
                        Clear). A model that always answers the same thing
                        scores 0 here however good its accuracy looks.

The second half asks the question left open since the ensemble work: if models
carry the surface-concreteness bias to different degrees, is a model with a
weaker dose worth adding to one with a stronger dose? The fine-tuned transformer
and the zero-shot open model are combined four ways, every weight fitted
out-of-fold and applied once to test.

CPU only; reads cached predictions from scripts 26, 36 and 44.
"""
import csv
import io
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
S26 = os.path.join(ROOT, "research", "artifacts", "s26")
S36 = os.path.join(ROOT, "research", "artifacts", "s36")
S44 = os.path.join(ROOT, "research", "artifacts", "s44")
os.makedirs(OUT, exist_ok=True)

SEED = 20260829
EQ = ["Clear", "Not Clear", "Misleading"]


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def gap(pred, y):
    """How much more often does it say Clear when the answer is Clear?"""
    a = [p == 0 for p, g in zip(pred, y) if g == 0]
    b = [p == 0 for p, g in zip(pred, y) if g == 1]
    return (np.mean(a) if a else float("nan")) - (np.mean(b) if b else float("nan"))


def row(name, pred, y, rng, ref=None, n_boot=2000, n_rows=None):
    mf = f1_score(y, pred, average="macro", zero_division=0, labels=[0, 1, 2])
    ba = balanced_accuracy_score(y, pred)
    ac = float(np.mean(np.asarray(pred) == np.asarray(y)))
    d = da = ""
    if ref is not None:
        y_, p_, r_ = np.asarray(y), np.asarray(pred), np.asarray(ref)
        bs = []
        for _ in range(n_boot):
            k = rng.integers(0, len(y_), len(y_))
            bs.append(f1_score(y_[k], p_[k], average="macro", zero_division=0,
                               labels=[0, 1, 2])
                      - f1_score(y_[k], r_[k], average="macro", zero_division=0,
                                 labels=[0, 1, 2]))
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        ab = [float((p_[k] == y_[k]).mean() - (r_[k] == y_[k]).mean())
              for k in (rng.integers(0, len(y_), len(y_)) for _ in range(n_boot))]
        alo, ahi = np.percentile(ab, 2.5), np.percentile(ab, 97.5)
        d = "%+.4f [%+.4f, %+.4f]" % (
            mf - f1_score(y, ref, average="macro", zero_division=0,
                          labels=[0, 1, 2]), lo, hi)
        da = "%+.4f [%+.4f, %+.4f]" % (
            ac - float((r_ == y_).mean()), alo, ahi)
    return "| %s | %s | %.4f | %.4f | %.4f | %+.3f | %s | %s |" % (
        name, "%d" % (n_rows if n_rows is not None else len(y)),
        mf, ba, ac, gap(pred, y), d or "--",
        (da if ref is not None else "--"))


def main():
    rng = np.random.default_rng(SEED)
    trv = [r for r in load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
           if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ]
    te = [r for r in load("vpesg4k_test_2000.csv")
          if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ]
    y_tr = np.array([EQ.index(na(r["evidence_quality"])) for r in trv])
    y_te = np.array([EQ.index(na(r["evidence_quality"])) for r in te])

    z = np.load(os.path.join(S36, "cond_A.npz"))
    tf_oof, tf_te = z["oof"], z["test"]   # already softmax probabilities
    qb = json.load(open(os.path.join(S44, "base_Qwen_Qwen3-8B.json"),
                        encoding="utf-8"))
    qw_oof, qw_te = softmax(qb["train"]), softmax(qb["test"])
    assert len(qw_oof) == len(y_tr) and len(qw_te) == len(y_te), "row counts differ"

    # Script 26 allowed N/A as an option and the frontier models used it on rows
    # that do have evidence. Such a prediction is simply wrong, so it is mapped
    # to an out-of-set index that scores as an error against every real class.
    def idx(lab):
        return EQ.index(lab) if lab in EQ else 3

    frontier = {}
    for f in sorted(os.listdir(S26)):
        if f.endswith(".json"):
            g = json.load(open(os.path.join(S26, f), encoding="utf-8"))
            frontier[f.split("__")[0]] = {k: (v[0] or "") for k, v in g.items()}

    L = []
    A = L.append
    A("# The comparison on rows nobody selected")
    A("")
    A("Generated by `research/scripts/45_unselected_comparison.py`. All %d test "
      "rows with `evidence_status=Yes`; no row here was chosen using any "
      "model's output." % len(te))
    A("")
    A("`discrimination` is P(says Clear | gold Clear) minus P(says Clear | gold "
      "Not Clear). It is the column that catches a model which has stopped "
      "answering the question: on this test set %.0f%% of the labels are "
      "`Clear`, so always answering `Clear` scores %.3f accuracy and 0.000 here."
      % (100 * np.mean(y_te == 0), np.mean(y_te == 0)))
    A("")
    A("| system | rows | macro-F1 | balanced acc. | accuracy | discrimination | macro-F1 vs fine-tuned | accuracy vs fine-tuned |")
    A("|---|---:|---:|---:|---:|---:|---|---|")
    ref = tf_te.argmax(1)
    A(row("fine-tuned transformer (script 36, condition A)", ref, y_te, rng))

    # a call that did not return is not an answer: that row leaves this model's
    # evaluation, and its own reference column is recomputed on the same subset
    def usable(m):
        return np.array([bool(frontier[m].get(r["id"], "")) for r in te])

    for m in sorted(frontier):
        k = usable(m)
        p = np.array([idx(frontier[m].get(r["id"], "")) for r in te])[k]
        A(row("%s, zero-shot" % m, p, y_te[k], rng, ref[k]))
    A(row("Qwen3-8B, zero-shot", qw_te.argmax(1), y_te, rng, ref))

    # the vote needs every member to have answered, so it is scored on the
    # intersection
    allk = np.ones(len(te), bool)
    for m in frontier:
        allk &= usable(m)
    votes = np.zeros((int(allk.sum()), 4))
    sub_te = [r for r, keep in zip(te, allk) if keep]
    for m in frontier:
        for n, r in enumerate(sub_te):
            votes[n, idx(frontier[m][r["id"]])] += 1
    A(row("four frontier models, majority vote", votes.argmax(1), y_te[allk],
          rng, ref[allk]))
    # the vote has to beat its own best member on the vote's own rows
    best = max(frontier, key=lambda m: f1_score(
        y_te[allk],
        np.array([idx(frontier[m][r["id"]]) for r in sub_te]),
        average="macro", zero_division=0, labels=[0, 1, 2]))
    A(row("...its best single member (%s) on those same rows" % best,
          np.array([idx(frontier[best][r["id"]]) for r in sub_te]),
          y_te[allk], rng, ref[allk]))
    A("")
    A("**Rows differ by model on purpose.** `gemini-3.7-flash` returned HTTP 503 "
      "on %d of these rows; those rows are excluded from its evaluation and from "
      "the vote, and the fine-tuned reference is recomputed on the same subset "
      "so each comparison stays paired. Scoring a failed call as `Clear` would "
      "hand it a correct answer on a set that is %.1f%% `Clear`; scoring it as "
      "an error would punish it twice."
      % (len(te) - int(allk.sum()), 100 * float(np.mean(y_te == 0))))
    A("")

    A("## Is a differently-biased model worth adding?")
    A("")
    A("The fine-tuned transformer and the zero-shot open model, combined four "
      "ways. Every weight is fitted on the %d out-of-fold train+val rows and "
      "applied once to test, so none of these is tuned on the numbers it "
      "reports." % len(trv))
    A("")
    best_w, best_s = 0.0, -1.0
    for w in np.arange(0, 1.01, 0.05):
        s = f1_score(y_tr, ((1 - w) * tf_oof + w * qw_oof).argmax(1),
                     average="macro", zero_division=0, labels=[0, 1, 2])
        if s > best_s:
            best_s, best_w = s, float(w)
    stack = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        np.hstack([tf_oof, qw_oof]), y_tr)

    A("| combination | macro-F1 | balanced acc. | accuracy | discrimination | macro-F1 vs alone | accuracy vs alone |")
    A("|---|---:|---:|---:|---:|---|---|")
    A(row("fine-tuned transformer alone", ref, y_te, rng))
    A(row("Qwen3-8B alone", qw_te.argmax(1), y_te, rng, ref))
    A(row("unweighted average of the two", (0.5 * tf_te + 0.5 * qw_te).argmax(1),
          y_te, rng, ref))
    A(row("weighted average, weight %.2f fitted out-of-fold" % best_w,
          ((1 - best_w) * tf_te + best_w * qw_te).argmax(1), y_te, rng, ref))
    A(row("stacked, fitted out-of-fold",
          stack.predict(np.hstack([tf_te, qw_te])), y_te, rng, ref))
    A("")
    A("The out-of-fold search picked weight **%.2f** for the open model "
      "(out-of-fold macro-F1 %.4f)." % (best_w, best_s))
    A("")

    A("## Reading")
    A("")
    A("**The first table reverses what script 43 appeared to show.** On rows "
      "nobody selected, the open model has the *weakest* discrimination of "
      "every system here -- +0.172 against +0.384 to +0.467 for the frontier "
      "models and +0.461 for the fine-tuned transformer -- and its macro-F1 is "
      "below all four. Its apparent advantage on the disputed rows was the "
      "selection, not the model. The corrected claim is narrow: the open model "
      "reproduces the *substitution* on a within-model comparison, which is "
      "what script 43 was built to test, and nothing about its relative "
      "accuracy survives.")
    A("")
    A("**The second table is a genuine gain, and it is not on macro-F1.** "
      "Averaging the fine-tuned transformer with the zero-shot open model "
      "raises accuracy from 0.8052 to 0.8300 with a bootstrap interval that "
      "excludes zero, while macro-F1 moves +0.0087 with an interval that does "
      "not. The out-of-fold search put weight 0.45 on a model that is worse "
      "than the transformer on every column, which only pays if the two fail "
      "on different rows -- so the bias is present in both but not to the same "
      "degree, and the difference is worth something.")
    A("")
    A("This is the third time in this project that an intervention improves "
      "the decisions a user would see while macro-F1 declines to record it. "
      "The first was forcing the label hierarchy to be consistent; the second "
      "was criterion-based abstention. A measure that gives a class with three "
      "test instances the same weight as one with 1,106 will keep producing "
      "this disagreement, and it is a property of the measure rather than of "
      "the interventions.")
    A("")
    dest = os.path.join(OUT, "unselected_comparison.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
