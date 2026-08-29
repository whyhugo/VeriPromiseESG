"""Does judging the span beat judging the paragraph, once the judge is strong enough?

Hypothesis B in the main report asked whether `evidence_quality` should be
decided on the evidence sentence rather than the whole paragraph. Script 07
answered no -- conditions B and C beat the full paragraph by +0.008 and +0.009
macro-F1, with bootstrap intervals crossing zero -- but that script used a
character n-gram TF-IDF classifier and said so in its own limitations: the probe
scored 0.2394 on `Not Clear` against 0.4296 for the transformer ensemble, so it
was weakest exactly where a span effect would have to appear. The conclusion was
recorded as unresolved.

Two things have changed since. Script 32 built a clause tagger that extracts
both spans well enough to use (promise char-F1 0.5903, evidence 0.6666), so the
question can be asked with *predicted* spans rather than gold ones, which is
what a deployed system would have. And script 34 found that four cheap features
computed on the spans -- promise-evidence overlap, novel content, length, length
ratio -- reproduce the gold label at 0.675 balanced accuracy, close to what
paragraph-level concreteness cues manage. That says there is signal on the spans
that a bag of character n-grams cannot represent.

So this re-runs the comparison with the same transformer backbone used
everywhere else in the project, over six input conditions:

  A  the full paragraph                     the current formulation
  B  the gold evidence span                 the reformulation, oracle version
  C  gold promise + gold evidence           both spans, oracle version
  D  a length-matched random substring      control: is B just shorter input?
  E  the predicted evidence span            the reformulation, deployable
  F  predicted promise + predicted evidence  both spans, deployable

D is the comparison that decides whether B means anything. E and F are the ones
that decide whether the reformulation is usable, since a system cannot be handed
gold spans at inference. Predicted spans come from script 32's saved clause
probabilities: out-of-fold for the training rows, fold-averaged for test, with
the decision threshold refitted here on out-of-fold data only.

Five-fold cross-validation on train+val, test scored once per condition.
"""
import argparse
import csv
import importlib.util
import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
S32 = os.path.join(ROOT, "research", "artifacts", "s32")
CACHE = os.path.join(ROOT, "research", "artifacts", "s36")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEED = 20260829
BACKBONE = "hfl/chinese-roberta-wwm-ext-large"
MAXLEN = 384
EQ = ["Clear", "Not Clear", "Misleading"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Clause segmentation is imported from script 32 rather than copied, so the
# spans decoded here are identical to the ones that script scored. Script 32
# rebinds sys.stdout the same way this one does; holding a reference to the
# wrapper we made keeps it from being collected and closing the shared buffer.
_keep_ours = sys.stdout
_spec = importlib.util.spec_from_file_location(
    "s32", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "32_clause_tagger.py"))
s32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s32)
_keep_theirs = sys.stdout       # both stay referenced, so neither closes stdout


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def gold_span_text(row, field):
    """The annotated span, discontinuous pieces joined in reading order."""
    return "，".join(s32.pieces_of(row.get(field)))


def decode_span(para, probs, thr):
    """Clauses above threshold, concatenated in reading order."""
    cl = s32.clauses_of(para)[:s32.MAX_CLAUSES]
    keep = [para[a:b] for n, (a, b) in enumerate(cl) if probs[n] > thr]
    return "".join(keep).strip()


def fit_threshold(rows, oof, field):
    """Refit the clause threshold on out-of-fold predictions only."""
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.05):
        f = s32.score(rows, oof, field, float(t))[2]
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def predicted_spans(field):
    """Map row id -> predicted span text, for train+val (OOF) and test."""
    p = os.path.join(S32, "%s.npz" % field)
    if not os.path.exists(p):
        return None, None, None
    z = np.load(p)
    trv = [r for r in load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
           if s32.ok(r.get(field))]
    te = [r for r in load("vpesg4k_test_2000.csv") if s32.ok(r.get(field))]
    thr, oof_f1 = fit_threshold(trv, z["oof"], field)
    out = {}
    for rows, probs in ((trv, z["oof"]), (te, z["test"])):
        for n, r in enumerate(rows):
            out[r["id"]] = decode_span(r["data"] or "", probs[n], thr)
    return out, thr, oof_f1


def build_inputs(rows, cond, pred_pr, pred_ev, rng):
    """The text handed to the classifier under each condition."""
    txt, empty = [], 0
    for r in rows:
        para = r["data"] or ""
        if cond == "A":
            s = para
        elif cond == "B":
            s = gold_span_text(r, "evidence_string")
        elif cond == "C":
            s = (gold_span_text(r, "promise_string") + "。"
                 + gold_span_text(r, "evidence_string"))
        elif cond == "D":
            n = len(gold_span_text(r, "evidence_string"))
            i = int(rng.integers(0, max(1, len(para) - n + 1)))
            s = para[i:i + n]
        elif cond == "E":
            s = (pred_ev or {}).get(r["id"], "")
        else:
            s = ((pred_pr or {}).get(r["id"], "") + "。"
                 + (pred_ev or {}).get(r["id"], ""))
        s = s.strip("。 ")
        if not s:
            empty += 1
            s = para          # what a deployed system would fall back to
        txt.append(s)
    return txt, empty


class DS(Dataset):
    """Holds raw text; padding happens per batch so the span conditions, whose
    inputs are a fraction of a paragraph, do not pay the full-length cost."""

    def __init__(self, txt, y, tok):
        self.txt, self.y, self.tok = txt, y, tok

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, i):
        return self.txt[i], self.y[i]


def collate(batch, tok):
    txt, y = zip(*batch)
    e = tok(list(txt), truncation=True, max_length=MAXLEN, padding=True,
            return_tensors="pt")
    return dict(e), torch.tensor(y, dtype=torch.long)


class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BACKBONE).float()
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.enc.config.hidden_size, len(EQ))

    def forward(self, enc):
        h = self.enc(**enc).last_hidden_state[:, 0]
        return self.head(self.drop(h))


def run_fold(tr_txt, tr_y, evals, epochs, bs, lr):
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    m = Clf().to(DEV)
    coll = lambda b: collate(b, tok)                       # noqa: E731
    dl = DataLoader(DS(tr_txt, tr_y, tok), batch_size=bs, shuffle=True,
                    drop_last=False, collate_fn=coll)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    n_steps = max(1, epochs * len(dl))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps,
                                              pct_start=0.1)
    w = torch.tensor([1.0 / max(1, tr_y.count(k)) for k in range(len(EQ))],
                     device=DEV, dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum() * len(EQ))
    m.train()
    for _ in range(epochs):
        for enc, y in dl:
            enc = {k: v.to(DEV) for k, v in enc.items()}
            opt.zero_grad()
            loss = lossf(m(enc), y.to(DEV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
    m.eval()
    outs = []
    with torch.no_grad():
        for txt in evals:
            pr = []
            for enc, _ in DataLoader(DS(txt, [0] * len(txt), tok), batch_size=32,
                                     collate_fn=coll):
                enc = {k: v.to(DEV) for k, v in enc.items()}
                pr.append(torch.softmax(m(enc), -1).cpu().numpy())
            outs.append(np.concatenate(pr) if pr else np.zeros((0, len(EQ))))
    del m
    torch.cuda.empty_cache()
    return outs


def boot_delta(y, pa, pb, rng, n=4000):
    y = np.asarray(y)
    d = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k])) < 2:
            continue
        d.append(f1_score(y[k], pb[k], average="macro", zero_division=0)
                 - f1_score(y[k], pa[k], average="macro", zero_division=0))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="*",
                    default=["A", "B", "C", "D", "E", "F"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    trv = [r for r in load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
           if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ]
    te = [r for r in load("vpesg4k_test_2000.csv")
          if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ]
    ytr = [EQ.index(na(r["evidence_quality"])) for r in trv]
    yte = [EQ.index(na(r["evidence_quality"])) for r in te]
    print("train+val %d, test %d" % (len(trv), len(te)), flush=True)

    pred_pr, thr_pr, f1_pr = predicted_spans("promise_string")
    pred_ev, thr_ev, f1_ev = predicted_spans("evidence_string")
    if pred_ev is None:
        print("no script 32 artifacts; conditions E and F unavailable", flush=True)
        args.conds = [c for c in args.conds if c not in ("E", "F")]
    else:
        print("clause thresholds: promise %.2f (OOF char-F1 %.4f), "
              "evidence %.2f (OOF char-F1 %.4f)"
              % (thr_pr, f1_pr, thr_ev, f1_ev), flush=True)

    res = {}
    for cond in args.conds:
        cf = os.path.join(CACHE, "cond_%s.npz" % cond)
        if os.path.exists(cf):
            z = np.load(cf)
            res[cond] = (z["oof"], z["test"], int(z["empty"]))
            print("[%s] cached" % cond, flush=True)
            continue
        rng = np.random.default_rng(SEED)
        tr_txt, e1 = build_inputs(trv, cond, pred_pr, pred_ev, rng)
        te_txt, e2 = build_inputs(te, cond, pred_pr, pred_ev, rng)
        print("[%s] mean chars train %.0f test %.0f, empty-span fallbacks %d"
              % (cond, np.mean([len(t) for t in tr_txt]),
                 np.mean([len(t) for t in te_txt]), e1 + e2), flush=True)
        oof = np.zeros((len(trv), len(EQ)))
        tp = np.zeros((len(te), len(EQ)))
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        for k, (ia, ib) in enumerate(kf.split(trv)):
            pv, pt = run_fold([tr_txt[i] for i in ia], [ytr[i] for i in ia],
                              [[tr_txt[i] for i in ib], te_txt],
                              args.epochs, args.bs, args.lr)
            oof[ib] = pv
            tp += pt / args.folds
            print("  [%s] fold %d/%d OOF macro-F1 %.4f"
                  % (cond, k + 1, args.folds,
                     f1_score([ytr[i] for i in ib], pv.argmax(1),
                              average="macro", zero_division=0)), flush=True)
        np.savez(cf, oof=oof, test=tp, empty=e1 + e2)
        res[cond] = (oof, tp, e1 + e2)

    # --- report --------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    NAMES = {
        "A": "full paragraph (current formulation)",
        "B": "gold evidence span",
        "C": "gold promise + gold evidence span",
        "D": "length-matched random substring (control)",
        "E": "predicted evidence span",
        "F": "predicted promise + predicted evidence span",
    }
    L = []
    A = L.append
    A("# Locate then judge, with a transformer instead of a probe")
    A("")
    A("Generated by `research/scripts/36_locate_then_judge.py`. Backbone `%s`, "
      "%d-fold cross-validation on the %d train+val rows with "
      "`evidence_status=Yes`, the %d test rows scored once per condition. "
      "Conditions E and F use spans predicted by script 32's clause tagger, "
      "out-of-fold for training rows and fold-averaged for test; the clause "
      "threshold is refitted here on out-of-fold data only."
      % (BACKBONE, args.folds, len(trv), len(te)))
    A("")
    A("## Test set")
    A("")
    A("| condition | input | macro-F1 | Clear | Not Clear | Misleading | "
      "balanced acc. | vs A | 95% CI |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---|")
    base = res.get("A")
    for c in args.conds:
        if c not in res:
            continue
        pred = res[c][1].argmax(1)
        per = f1_score(yte, pred, average=None,
                       labels=list(range(len(EQ))), zero_division=0)
        mf = f1_score(yte, pred, average="macro", zero_division=0)
        ba = balanced_accuracy_score(yte, pred)
        if base is not None and c != "A":
            lo, hi = boot_delta(yte, base[1].argmax(1), pred, rng)
            d = "%+.4f" % (mf - f1_score(yte, base[1].argmax(1),
                                         average="macro", zero_division=0))
            ci = "[%+.4f, %+.4f]" % (lo, hi)
        else:
            d, ci = "--", "--"
        A("| %s | %s | **%.4f** | %.4f | %.4f | %.4f | %.4f | %s | %s |"
          % (c, NAMES[c], mf, per[0], per[1], per[2], ba, d, ci))
    A("")
    A("## Out-of-fold (train+val)")
    A("")
    A("| condition | macro-F1 | balanced acc. | empty-span fallbacks |")
    A("|---|---:|---:|---:|")
    for c in args.conds:
        if c not in res:
            continue
        pred = res[c][0].argmax(1)
        A("| %s | %.4f | %.4f | %d |"
          % (c, f1_score(ytr, pred, average="macro", zero_division=0),
             balanced_accuracy_score(ytr, pred), res[c][2]))
    A("")
    A("## Reading")
    A("")
    A("The comparison that matters is B against D, not B against A. If handing "
      "the classifier the gold evidence span helps by about as much as handing "
      "it a random substring of the same length, the gain is from shorter input "
      "and the reformulation is not doing the work.")
    A("")
    A("E and F are the deployable versions. A gap between C and F measures what "
      "the extractor costs; a gap between A and F measures what the whole "
      "locate-then-judge pipeline is worth to a system that has to find its own "
      "spans.")
    A("")
    json.dump({c: {"test_macro_f1": float(f1_score(yte, res[c][1].argmax(1),
                                                   average="macro",
                                                   zero_division=0))}
               for c in res}, open(os.path.join(OUT, "locate_then_judge.json"),
                                   "w", encoding="utf-8"), indent=1)
    dest = os.path.join(OUT, "locate_then_judge.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
