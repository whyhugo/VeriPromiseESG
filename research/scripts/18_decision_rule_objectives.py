"""Different metrics want different systems. How different?

Script 16 turned up something awkward: the submission we picked as our best is
first under the competition formula and *last* under joint exact match. The
difference between those two submissions was per-class threshold tuning against
macro-F1. So the decision rule -- not the representation -- was what the metric
choice changed.

That is worth measuring properly rather than noting in passing. Here the model's
probabilities are held completely fixed and only the decision rule varies: each
subtask gets a class-wise multiplicative prior, predict argmax(w_c * p_c), and w
is fitted by coordinate ascent against one objective at a time.

Four objectives:

    competition   0.20*PS + 0.30*ES + 0.35*EQ + 0.15*VT, macro-F1
    uniform       unweighted mean of the four macro-F1 scores
    exact         fraction of paragraphs where all four fields are right
    mcc           unweighted mean of the four Matthews correlations

Two things come out. Cross-evaluating every fitted rule under every objective
says how much the objectives disagree; and since `exact` couples the four
subtasks while the others do not, fitting it directly is the only way to get a
rule that trades one field against another.

Both an honest and an oracle version are reported. The honest one fits w on
out-of-fold predictions and evaluates once on test. The oracle one fits on test
directly -- not a performance claim, but an upper bound on how far apart the
objectives can pull, which is the quantity the argument actually needs.

Also fits a rule for `evidence_status` alone, since script 17 identified recall
0.514 on its `No` class as the one reachable, unattempted lever left.
"""
import argparse
import csv
import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
SUBS = os.path.join(ROOT, "submissions")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s18")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824

FIELDS = ["promise_status", "evidence_status", "evidence_quality",
          "verification_timeline"]
COMP_W = {"promise_status": 0.20, "evidence_status": 0.30,
          "evidence_quality": 0.35, "verification_timeline": 0.15}
LABELS = {
    "promise_status": ["No", "Yes"],
    "evidence_status": ["N/A", "No", "Yes"],
    "evidence_quality": ["N/A", "Not Clear", "Clear", "Misleading"],
    "verification_timeline": ["N/A", "already", "within_2_years",
                              "between_2_and_5_years", "more_than_5_years"],
}
BACKBONE = "hfl/chinese-roberta-wwm-ext-large"


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


class DS(Dataset):
    def __init__(self, texts, ys, tok, maxlen):
        self.t, self.y, self.tok, self.m = texts, ys, tok, maxlen

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, max_length=self.m,
                     padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(self.y[i])


def run_fold(tr_txt, tr_y, eval_txts, nlab, maxlen, epochs, bs, lr):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=nlab).to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(DS(tr_txt, tr_y, tok, maxlen), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, len(dl) * epochs), pct_start=0.1)
    cnt = np.bincount(tr_y, minlength=nlab).astype(float)
    w = torch.tensor(len(tr_y) / (nlab * np.maximum(cnt, 1)),
                     dtype=torch.float32, device="cuda")
    lossf = nn.CrossEntropyLoss(weight=w)
    mdl.train()
    for _ in range(epochs):
        for enc, y in dl:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            y = y.to("cuda")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(mdl(**enc).logits.float(), y)
            loss.backward()
            opt.step()
            sched.step()
    mdl.eval()
    outs = []
    for txts in eval_txts:
        dle = DataLoader(DS(txts, [0] * len(txts), tok, maxlen), batch_size=bs * 2)
        ps = []
        with torch.no_grad():
            for enc, _ in dle:
                enc = {k: v.to("cuda") for k, v in enc.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = mdl(**enc).logits.float()
                ps.append(torch.softmax(lg, -1).cpu().numpy())
        outs.append(np.concatenate(ps))
    del mdl
    torch.cuda.empty_cache()
    return outs


def get_probs(args):
    """5-fold CV producing out-of-fold and test probabilities per subtask."""
    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    te = load("vpesg4k_test_2000.csv")
    te_txt = [r["data"] or "" for r in te]
    oof_all, test_all, gold_oof = {}, {}, {}
    for f in FIELDS:
        cf = os.path.join(CACHE, "%s.npz" % f.replace("_", ""))
        if os.path.exists(cf):
            z = np.load(cf, allow_pickle=True)
            oof_all[f], test_all[f], gold_oof[f] = z["oof"], z["test"], z["y"]
            print("[cached]", f, flush=True)
            continue
        labels = LABELS[f]
        pool = [r for r in trv if na(r[f]) in labels]
        txt = [r["data"] or "" for r in pool]
        y = np.array([labels.index(na(r[f])) for r in pool])
        oof = np.zeros((len(pool), len(labels)))
        tp = np.zeros((len(te), len(labels)))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for k, (ia, ib) in enumerate(skf.split(np.zeros(len(pool)), y)):
            pv, pt = run_fold([txt[i] for i in ia], y[ia],
                              [[txt[i] for i in ib], te_txt], len(labels),
                              args.maxlen, args.epochs, args.bs, args.lr)
            oof[ib] = pv
            tp += pt / 5
            print("  [%s] fold %d/5" % (f, k + 1), flush=True)
        np.savez(cf, oof=oof, test=tp, y=y)
        oof_all[f], test_all[f], gold_oof[f] = oof, tp, y
    return oof_all, test_all, gold_oof, trv, te


def predict(probs, w):
    return {f: (probs[f] * w[f]).argmax(1) for f in FIELDS}


def objective(name, pred_idx, gold_idx):
    per = {}
    for f in FIELDS:
        y, p = gold_idx[f], pred_idx[f]
        if name == "mcc":
            per[f] = matthews_corrcoef(y, p)
        else:
            per[f] = f1_score(y, p, average="macro", zero_division=0)
    if name == "competition":
        return sum(COMP_W[f] * per[f] for f in FIELDS)
    if name in ("uniform", "mcc"):
        return float(np.mean([per[f] for f in FIELDS]))
    if name == "exact":
        ok = np.ones(len(gold_idx[FIELDS[0]]), dtype=bool)
        for f in FIELDS:
            ok &= (pred_idx[f] == gold_idx[f])
        return float(ok.mean())
    raise ValueError(name)


def fit_rule(probs, gold_idx, name, rounds=6, grid=None):
    grid = grid if grid is not None else np.exp(np.linspace(-1.6, 1.6, 17))
    w = {f: np.ones(probs[f].shape[1]) for f in FIELDS}
    best = objective(name, predict(probs, w), gold_idx)
    for _ in range(rounds):
        improved = False
        for f in FIELDS:
            for c in range(probs[f].shape[1]):
                cur = w[f][c]
                for g in grid:
                    w[f][c] = g
                    v = objective(name, predict(probs, w), gold_idx)
                    if v > best + 1e-9:
                        best, cur, improved = v, g, True
                w[f][c] = cur
        if not improved:
            break
    return w, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    oof, test, gold_oof, trv, te = get_probs(args)
    gold_te = {f: np.array([LABELS[f].index(na(r[f])) for r in te]) for f in FIELDS}
    # the out-of-fold pools differ in length per subtask, so tune each subtask on
    # its own pool for the decomposable objectives; `exact` needs one shared pool
    common = {f: np.array([LABELS[f].index(na(r[f])) for r in trv
                           if na(r[f]) in LABELS[f]]) for f in FIELDS}
    sizes = {f: len(common[f]) for f in FIELDS}
    shared_ok = len(set(sizes.values())) == 1

    OBJ = ["competition", "uniform", "exact", "mcc"]
    L = []
    A = L.append
    A("# What the choice of objective does to the decision rule")
    A("")
    A("Generated by `research/scripts/18_decision_rule_objectives.py`. Encoder "
      "`%s`, 5-fold CV, probabilities held fixed; only a class-wise "
      "multiplicative prior varies, fitted by coordinate ascent." % BACKBONE)
    A("")
    A("Out-of-fold pool sizes: %s.%s"
      % (", ".join("%s %d" % (f, sizes[f]) for f in FIELDS),
         "" if shared_ok else " These differ, so `exact` is fitted on test only "
         "and reported as an oracle bound."))
    A("")

    A("## Oracle: how far apart can the objectives pull?")
    A("")
    A("Each rule is fitted directly on test against one objective, then scored "
      "under all four. Not a performance claim -- an upper bound on the "
      "disagreement between objectives.")
    A("")
    rules, diag = {}, {}
    for name in OBJ:
        w, v = fit_rule(test, gold_te, name)
        rules[name] = w
        diag[name] = v
        print("fitted oracle rule for", name, "%.4f" % v, flush=True)
    A("| fitted for \\ scored as | %s |" % " | ".join(OBJ))
    A("|---|%s" % ("---:|" * len(OBJ)))
    base_w = {f: np.ones(test[f].shape[1]) for f in FIELDS}
    row = ["%.4f" % objective(o, predict(test, base_w), gold_te) for o in OBJ]
    A("| **no adjustment (argmax)** | %s |" % " | ".join(row))
    for name in OBJ:
        row = ["%.4f" % objective(o, predict(test, rules[name]), gold_te)
               for o in OBJ]
        row[OBJ.index(name)] = "**%s**" % row[OBJ.index(name)]
        A("| fitted for `%s` | %s |" % (name, " | ".join(row)))
    A("")
    ce = objective("exact", predict(test, rules["competition"]), gold_te)
    ee = objective("exact", predict(test, rules["exact"]), gold_te)
    cc = objective("competition", predict(test, rules["competition"]), gold_te)
    ec = objective("competition", predict(test, rules["exact"]), gold_te)
    A("The two corners of the trade: optimising the competition objective "
      "reaches %.4f on it while giving up %.4f on exact match; optimising exact "
      "match reaches %.4f while giving up %.4f on the competition objective. "
      "The rules are not the same rule." % (cc, ee - ce, ee, cc - ec))
    A("")

    A("## Honest: fitted out-of-fold, evaluated once on test")
    A("")
    A("Decomposable objectives only (`exact` couples the four subtasks and "
      "their out-of-fold pools are not aligned).")
    A("")
    A("| rule | competition | uniform | exact | mcc |")
    A("|---|---:|---:|---:|---:|")
    row = ["%.4f" % objective(o, predict(test, base_w), gold_te) for o in OBJ]
    A("| no adjustment (argmax) | %s |" % " | ".join(row))
    for name in ["competition", "uniform", "mcc"]:
        w, _ = fit_rule(oof, gold_oof, name)
        row = ["%.4f" % objective(o, predict(test, w), gold_te) for o in OBJ]
        A("| fitted out-of-fold for `%s` | %s |" % (name, " | ".join(row)))
        json.dump({f: list(map(float, w[f])) for f in FIELDS},
                  open(os.path.join(CACHE, "rule_%s.json" % name), "w"), indent=1)
    A("")

    A("## The `evidence_status = No` lever")
    A("")
    A("Script 17 found recall 0.514 on this class with 282 training instances, "
      "the one reachable and unattempted piece of headroom. Sweeping only its "
      "prior, everything else fixed:")
    A("")
    A("| prior on `No` | recall on `No` | `evidence_status` macro-F1 | competition | exact |")
    A("|---:|---:|---:|---:|---:|")
    es = "evidence_status"
    ino = LABELS[es].index("No")
    for g in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        w = {f: np.ones(test[f].shape[1]) for f in FIELDS}
        w[es][ino] = g
        pi = predict(test, w)
        yn = gold_te[es] == ino
        rec = float((pi[es][yn] == ino).mean())
        A("| %.1f | %.4f | %.4f | %.4f | %.4f |"
          % (g, rec,
             f1_score(gold_te[es], pi[es], average="macro", zero_division=0),
             objective("competition", pi, gold_te),
             objective("exact", pi, gold_te)))
    A("")

    dest = os.path.join(OUT, "decision_rule_objectives.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
