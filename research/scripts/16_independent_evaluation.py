"""An evaluation framework that does not inherit the competition's weighting.

Everything reported so far has been scored with the official formula,

    0.20*PS + 0.30*ES + 0.35*EQ + 0.15*VT,   all macro-F1,

and several of our headline numbers are artifacts of it. "Misleading is worth
8.75 points" is just 0.35 x 0.25; under uniform weights it would be 6.25, under
support weighting almost nothing. A weighting chosen to shape a leaderboard is
not a defensible measuring instrument for a paper.

The three NTCIR-18 AI CUP special-session papers all take the same line: they
use the competition *dataset* as material and then define discipline-standard
metrics for their own research question -- MRR@10, NDCG@10, Recall@100, AP@1 --
together with an explicit baseline, an approximate upper bound, ablations, and
significance testing. None of them reports a leaderboard score as the paper's
evidence.

This script builds the equivalent for a hierarchical multi-task classification
problem:

  1. per-subtask reporting with several summary statistics, never collapsed
     into one weighted number by default
  2. metrics that behave sensibly under extreme imbalance: balanced accuracy,
     Matthews correlation, Cohen's kappa, alongside macro- and micro-F1
  3. joint exact match -- the fraction of paragraphs where all four fields are
     right, which is what an auditor actually needs and which penalises
     structural inconsistency without any invented index
  4. conditional (hierarchical) evaluation, scoring each subtask only on rows
     where it applies, to separate gating errors from judgement errors
  5. a human-ceiling normalisation: Krippendorff's alpha between prediction and
     gold, divided by the inter-annotator alpha the organisers report, giving
     "how far to the human ceiling" independent of any weighting
  6. bootstrap confidence intervals
  7. a weighting-sensitivity analysis that tests the critique directly: does
     the ranking of systems change under competition / uniform / support
     weighting?

Runs on CPU against any submission CSV.
"""
import argparse
import csv
import io
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
SUBS = os.path.join(ROOT, "submissions")
OUT = os.path.join(ROOT, "research", "output")
os.makedirs(OUT, exist_ok=True)
SEED = 20260824

FIELDS = ["promise_status", "evidence_status", "evidence_quality",
          "verification_timeline"]
COMP_W = {"promise_status": 0.20, "evidence_status": 0.30,
          "evidence_quality": 0.35, "verification_timeline": 0.15}
# inter-annotator agreement reported by the organisers
HUMAN_ALPHA = {"promise_status": 0.977, "evidence_status": 0.962,
               "evidence_quality": 0.940, "verification_timeline": 0.955}
# rows on which each subtask is actually applicable, per the guideline
APPLICABLE = {"promise_status": None,
              "evidence_status": ("promise_status", "Yes"),
              "verification_timeline": ("promise_status", "Yes"),
              "evidence_quality": ("evidence_status", "Yes")}


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def load_gold():
    return list(csv.DictReader(
        open(os.path.join(D, "vpesg4k_test_2000.csv"), encoding="utf-8-sig")))


def load_sub(fn):
    return {r["id"]: r for r in csv.DictReader(
        open(os.path.join(SUBS, fn), encoding="utf-8-sig"))}


def krippendorff_alpha_nominal(a, b):
    """Two-coder nominal alpha. Equivalent to the organisers' statistic."""
    units = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(units)
    if n == 0:
        return float("nan")
    do = sum(1.0 for x, y in units if x != y) / n           # observed disagreement
    vals = Counter()
    for x, y in units:
        vals[x] += 1
        vals[y] += 1
    N = 2 * n
    de = 1.0 - sum(c * (c - 1) for c in vals.values()) / (N * (N - 1))
    return 1.0 - do / de if de > 0 else float("nan")


def subtask_metrics(y, p):
    labels = sorted(set(y) | set(p))
    out = {
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "micro_f1": f1_score(y, p, average="micro", zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y, p),
        "balanced_acc": balanced_accuracy_score(y, p),
        "mcc": matthews_corrcoef(y, p),
        "kappa": cohen_kappa_score(y, p, labels=labels),
        "alpha": krippendorff_alpha_nominal(y, p),
    }
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, p, labels=labels, zero_division=0)
    out["per_class"] = {labels[i]: {"p": pr[i], "r": rc[i], "f1": f1[i],
                                    "support": int(sup[i])}
                        for i in range(len(labels))}
    return out


def joint_exact_match(gold, pm):
    ok = sum(1 for r in gold
             if all(na(pm[r["id"]][f]) == na(r[f]) for f in FIELDS))
    return ok / len(gold)


def conditional_metrics(gold, pm, field):
    """Score `field` only on rows where the gold upstream makes it applicable."""
    cond = APPLICABLE[field]
    rows = gold if cond is None else [r for r in gold if na(r[cond[0]]) == cond[1]]
    if not rows:
        return None
    y = [na(r[field]) for r in rows]
    p = [na(pm[r["id"]][field]) for r in rows]
    # Average only over labels the guideline permits in this condition. A
    # prediction outside that set still counts as a miss for its true class,
    # but must not add a spurious zero-support class to the macro average.
    labels = sorted(set(y))
    return (len(rows),
            f1_score(y, p, labels=labels, average="macro", zero_division=0),
            balanced_accuracy_score(y, p),
            sum(1 for v in p if v not in labels))


def bootstrap_ci(gold, pm, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(gold))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals.append(fn([gold[i] for i in s], pm))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summaries(gold, pm):
    per = {f: subtask_metrics([na(r[f]) for r in gold],
                              [na(pm[r["id"]][f]) for r in gold])
           for f in FIELDS}
    supp = {f: len(FIELDS) and 1.0 for f in FIELDS}
    n_classes = {f: len(per[f]["per_class"]) for f in FIELDS}
    return {
        "competition_weighted": sum(COMP_W[f] * per[f]["macro_f1"] for f in FIELDS),
        "uniform_macro_f1": float(np.mean([per[f]["macro_f1"] for f in FIELDS])),
        "uniform_balanced_acc": float(np.mean([per[f]["balanced_acc"] for f in FIELDS])),
        "uniform_mcc": float(np.mean([per[f]["mcc"] for f in FIELDS])),
        "uniform_weighted_f1": float(np.mean([per[f]["weighted_f1"] for f in FIELDS])),
        "joint_exact_match": joint_exact_match(gold, pm),
        "ceiling_ratio": float(np.mean([per[f]["alpha"] / HUMAN_ALPHA[f]
                                        for f in FIELDS])),
        "_per": per,
        "_nclass": n_classes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="*", default=[
        "ens_3way_tuned.csv", "ens_3way_raw.csv", "ens_rob_mac_tuned.csv",
        "roberta_large_cv5_ensemble_tuned.csv", "roberta_large_cv5_ensemble.csv"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    gold = load_gold()
    subs = {}
    for fn in args.subs:
        p = os.path.join(SUBS, fn)
        if os.path.exists(p):
            subs[fn] = load_sub(fn)
        else:
            print("skip missing", fn, flush=True)

    # trivial reference systems
    maj = {f: Counter(na(r[f]) for r in gold).most_common(1)[0][0] for f in FIELDS}
    subs["[reference] always-majority"] = {
        r["id"]: dict({"id": r["id"]}, **maj) for r in gold}
    rng = np.random.default_rng(SEED)
    pri = {f: (lambda c: (list(c.keys()), np.array(list(c.values()), float) / sum(c.values())))(
        Counter(na(r[f]) for r in gold)) for f in FIELDS}
    subs["[reference] stratified-random"] = {
        r["id"]: dict({"id": r["id"]},
                      **{f: str(rng.choice(pri[f][0], p=pri[f][1])) for f in FIELDS})
        for r in gold}

    res = {k: summaries(gold, v) for k, v in subs.items()}

    L = []
    A = L.append
    A("# An evaluation framework independent of the competition weighting")
    A("")
    A("Generated by `research/scripts/16_independent_evaluation.py` on the "
      "official 2,000-row test set.")
    A("")
    A("## Why this exists")
    A("")
    A("The official formula is `0.20*PS + 0.30*ES + 0.35*EQ + 0.15*VT`, all "
      "macro-F1. Those weights are a competition design choice, and some of our "
      "earlier claims inherit them: \"Misleading decides 8.75 points\" is just "
      "`0.35 x 0.25`. A weighting chosen to shape a leaderboard is not a "
      "measuring instrument for a paper.")
    A("")
    A("All three NTCIR-18 AI CUP special-session papers take the same line -- "
      "they use the competition dataset as material and then define "
      "discipline-standard metrics for their own question (MRR@10, NDCG@10, "
      "Recall@100, AP@1), with an explicit baseline, an approximate upper "
      "bound, ablations, and significance testing. None reports a leaderboard "
      "score as the paper's evidence.")
    A("")

    A("## Summary statistics under different aggregation choices")
    A("")
    A("| system | competition-weighted | uniform macro-F1 | uniform balanced acc. | uniform MCC | support-weighted F1 | joint exact match | human-ceiling ratio |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, r in sorted(res.items(), key=lambda kv: -kv[1]["competition_weighted"]):
        A("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f |"
          % (k, r["competition_weighted"], r["uniform_macro_f1"],
             r["uniform_balanced_acc"], r["uniform_mcc"],
             r["uniform_weighted_f1"], r["joint_exact_match"],
             r["ceiling_ratio"]))
    A("")
    A("`joint exact match` is the fraction of paragraphs where **all four** "
      "fields are correct -- what an auditor needs, and a quantity that "
      "penalises structural inconsistency without any invented index. "
      "`human-ceiling ratio` is Krippendorff's alpha between prediction and "
      "gold divided by the inter-annotator alpha the organisers report "
      "(0.977 / 0.962 / 0.940 / 0.955), averaged over subtasks: 1.0 would mean "
      "the model agrees with the gold labels as strongly as the annotators "
      "agreed with each other.")
    A("")

    A("## Does the choice of aggregation change the ranking?")
    A("")
    orders = {}
    for key in ["competition_weighted", "uniform_macro_f1", "uniform_balanced_acc",
                "uniform_mcc", "uniform_weighted_f1", "joint_exact_match"]:
        orders[key] = [k for k, _ in sorted(res.items(), key=lambda kv: -kv[1][key])]
    real = [k for k in orders["competition_weighted"] if not k.startswith("[reference]")]
    A("| aggregation | ranking of the real systems |")
    A("|---|---|")
    for key, order in orders.items():
        o = [k.replace(".csv", "") for k in order if not k.startswith("[reference]")]
        A("| `%s` | %s |" % (key, " > ".join(o)))
    A("")
    same = len({tuple(k.replace('.csv', '') for k in o if not k.startswith('[reference]'))
                for o in orders.values()}) == 1
    A("The ranking is **%s** across aggregation choices."
      % ("stable" if same else "not stable"))
    A("")

    A("## Per-subtask detail for the primary system")
    A("")
    primary = "ens_3way_tuned.csv"
    if primary in res:
        r = res[primary]["_per"]
        A("| subtask | classes | macro-F1 | balanced acc. | MCC | Cohen kappa | alpha vs gold | human alpha | ratio |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for f in FIELDS:
            m = r[f]
            A("| `%s` | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f | **%.3f** |"
              % (f, len(m["per_class"]), m["macro_f1"], m["balanced_acc"],
                 m["mcc"], m["kappa"], m["alpha"], HUMAN_ALPHA[f],
                 m["alpha"] / HUMAN_ALPHA[f]))
        A("")
        A("### Conditional (hierarchical) view")
        A("")
        A("Each subtask scored only on rows where the **gold** upstream label "
          "makes it applicable, which separates gating errors from judgement "
          "errors.")
        A("")
        A("| subtask | applicable rows | macro-F1 (conditional) | balanced acc. | macro-F1 (all rows) | impossible labels emitted |")
        A("|---|---:|---:|---:|---:|---:|")
        for f in FIELDS:
            c = conditional_metrics(gold, subs[primary], f)
            if c:
                A("| `%s` | %d | %.4f | %.4f | %.4f | %d |"
                  % (f, c[0], c[1], c[2], r[f]["macro_f1"], c[3]))
        A("")
        A("The last column counts predictions that the guideline makes "
          "impossible on those rows -- e.g. `N/A` for `evidence_status` on a "
          "paragraph that does contain a promise.")
        A("")
        A("### Per-class detail, `evidence_quality`")
        A("")
        A("| class | support | precision | recall | F1 |")
        A("|---|---:|---:|---:|---:|")
        for c, v in sorted(r["evidence_quality"]["per_class"].items(),
                           key=lambda kv: -kv[1]["support"]):
            A("| `%s` | %d | %.4f | %.4f | %.4f |"
              % (c, v["support"], v["p"], v["r"], v["f1"]))
        A("")

    A("## How much leverage does a 3-instance class really have?")
    A("")
    A("| aggregation | value with Misleading | value ignoring Misleading | difference |")
    A("|---|---:|---:|---:|")
    if primary in subs:
        y = [na(r["evidence_quality"]) for r in gold]
        p = [na(subs[primary][r["id"]]["evidence_quality"]) for r in gold]
        keep = [c for c in sorted(set(y)) if c != "Misleading"]
        for nm, kw in [("macro-F1", {"average": "macro"}),
                       ("support-weighted F1", {"average": "weighted"})]:
            a = f1_score(y, p, zero_division=0, **kw)
            b = f1_score(y, p, labels=keep, zero_division=0, **kw)
            A("| %s | %.4f | %.4f | %+.4f |" % (nm, a, b, b - a))
        idx = [i for i, v in enumerate(y) if v != "Misleading"]
        A("| balanced accuracy | %.4f | %.4f | %+.4f |"
          % (balanced_accuracy_score(y, p),
             balanced_accuracy_score([y[i] for i in idx], [p[i] for i in idx]),
             balanced_accuracy_score([y[i] for i in idx], [p[i] for i in idx])
             - balanced_accuracy_score(y, p)))
    A("")
    A("A class with 3 of 2,000 instances moves macro-F1 by a quarter of the "
      "subtask while moving support-weighted F1 and accuracy by almost nothing. "
      "That gap is the argument: macro-averaging is not defensible when a class "
      "is this degenerate, regardless of what weight the competition puts on "
      "the subtask.")
    A("")

    A("## Bootstrap confidence intervals")
    A("")
    A("2,000 test rows, %d resamples, percentile intervals." % args.boot)
    A("")
    A("| system | joint exact match | 95% CI | uniform macro-F1 | 95% CI |")
    A("|---|---:|---|---:|---|")
    for k in list(subs):
        if k.startswith("[reference] stratified"):
            continue
        lo1, hi1 = bootstrap_ci(gold, subs[k], joint_exact_match, args.boot)
        lo2, hi2 = bootstrap_ci(
            gold, subs[k],
            lambda g, m: float(np.mean([
                f1_score([na(r[f]) for r in g], [na(m[r["id"]][f]) for r in g],
                         average="macro", zero_division=0) for f in FIELDS])),
            args.boot)
        A("| %s | %.4f | [%.4f, %.4f] | %.4f | [%.4f, %.4f] |"
          % (k.replace(".csv", ""), res[k]["joint_exact_match"], lo1, hi1,
             res[k]["uniform_macro_f1"], lo2, hi2))
    A("")

    dest = os.path.join(OUT, "independent_evaluation.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
               for k, v in res.items()},
              open(os.path.join(OUT, "independent_evaluation.json"), "w"), indent=1)
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
