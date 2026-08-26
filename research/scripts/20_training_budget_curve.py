"""Was the fixed 4-epoch budget hiding the architecture comparison?

Scripts 13-15 all train a fixed 4 epochs. The competition system trained 15 with
best-epoch selection on a validation split. So when script 15's best new
configuration reached 0.6157 against the competition ensemble's 0.6384, that gap
mixed two things -- architecture and training budget -- and no architectural
conclusion can be drawn from it until they are separated.

Rather than run two arms, this trains once at 15 epochs and evaluates the
validation fold after *every* epoch, so the 4-epoch number and the 15-epoch
number both come out of the same run, along with the whole curve between them.

The curve answers a second question the paper needs. If `evidence_quality`
plateaus after a couple of epochs while `verification_timeline` keeps improving,
that is evidence about the *kind* of difficulty each subtask presents: a subtask
that saturates early is limited by what the input supports, not by optimisation.
That distinction is what "we could not improve subtask three" has to rest on if
it is going to be a finding rather than an admission.

One backbone, all four subtasks, 5 folds, full label set including N/A, so the
numbers line up directly with the `dedicated, 1 backbone` rows of script 15.
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s20")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824
BACKBONE = "hfl/chinese-roberta-wwm-ext-large"

FIELDS = [("promise_status", 0.20), ("evidence_status", 0.30),
          ("evidence_quality", 0.35), ("verification_timeline", 0.15)]
LABELS = {
    "promise_status": ["No", "Yes"],
    "evidence_status": ["N/A", "No", "Yes"],
    "evidence_quality": ["N/A", "Not Clear", "Clear", "Misleading"],
    "verification_timeline": ["N/A", "already", "within_2_years",
                              "between_2_and_5_years", "more_than_5_years"],
}


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


def infer(mdl, tok, texts, maxlen, bs):
    # 15 epochs means 15 inference passes over the 2,000-row test set per fold,
    # which is a large fraction of the run if the eval batch matches training.
    dle = DataLoader(DS(texts, [0] * len(texts), tok, maxlen), batch_size=bs * 4)
    ps = []
    mdl.eval()
    with torch.no_grad():
        for enc, _ in dle:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ps.append(torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy())
    mdl.train()
    return np.concatenate(ps)


def run_fold(tr_txt, tr_y, va_txt, va_y, te_txt, nlab, maxlen, epochs, bs, lr):
    """Train `epochs` epochs, scoring the validation fold after each one."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=nlab).float().to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(DS(tr_txt, tr_y, tok, maxlen), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, len(dl) * epochs), pct_start=0.1)
    cnt = np.bincount(tr_y, minlength=nlab).astype(float)
    w = torch.tensor(len(tr_y) / (nlab * np.maximum(cnt, 1)),
                     dtype=torch.float32, device="cuda")
    lossf = nn.CrossEntropyLoss(weight=w)

    curve, va_probs, te_probs = [], [], []
    mdl.train()
    for ep in range(epochs):
        for enc, y in dl:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            y = y.to("cuda")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(mdl(**enc).logits.float(), y)
            loss.backward()
            opt.step()
            sched.step()
        pv = infer(mdl, tok, va_txt, maxlen, bs)
        curve.append(f1_score(va_y, pv.argmax(1), average="macro", zero_division=0))
        va_probs.append(pv)
        te_probs.append(infer(mdl, tok, te_txt, maxlen, bs))
    del mdl
    torch.cuda.empty_cache()
    return np.array(curve), np.array(va_probs), np.array(te_probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fields", nargs="*", default=None,
                    help="subtasks to run; default all four")
    args = ap.parse_args()
    fields = ([(f, w) for f, w in FIELDS if f in args.fields]
              if args.fields else FIELDS)

    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    te = load("vpesg4k_test_2000.csv")
    te_txt = [r["data"] or "" for r in te]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    curves, oof_by_ep, test_by_ep, gold = {}, {}, {}, {}
    for f, _ in fields:
        cf = os.path.join(CACHE, "%s.npz" % f.replace("_", ""))
        if os.path.exists(cf):
            z = np.load(cf)
            curves[f], oof_by_ep[f] = z["curve"], z["oof"]
            test_by_ep[f], gold[f] = z["test"], z["y"]
            print("[cached]", f, flush=True)
            continue
        labels = LABELS[f]
        pool = [r for r in trv if na(r[f]) in labels]
        txt = [r["data"] or "" for r in pool]
        y = np.array([labels.index(na(r[f])) for r in pool])
        oof = np.zeros((args.epochs, len(pool), len(labels)))
        tp = np.zeros((args.epochs, len(te), len(labels)))
        cur = np.zeros((args.folds, args.epochs))
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        for k, (ia, ib) in enumerate(skf.split(np.zeros(len(pool)), y)):
            c, pv, pt = run_fold([txt[i] for i in ia], y[ia],
                                 [txt[i] for i in ib], y[ib], te_txt,
                                 len(labels), args.maxlen, args.epochs,
                                 args.bs, args.lr)
            cur[k] = c
            oof[:, ib, :] = pv
            tp += pt / args.folds
            print("  [%s] fold %d/%d  best epoch %d (%.4f)"
                  % (f, k + 1, args.folds, int(c.argmax()) + 1, c.max()), flush=True)
        np.savez(cf, curve=cur, oof=oof, test=tp, y=y)
        curves[f], oof_by_ep[f], test_by_ep[f], gold[f] = cur, oof, tp, y
        print("[%s] done" % f, flush=True)

    gold_te = {f: np.array([LABELS[f].index(na(r[f])) for r in te])
               for f, _ in fields}

    L = []
    A = L.append
    A("# Training-budget curve")
    A("")
    A("Generated by `research/scripts/20_training_budget_curve.py`. Encoder "
      "`%s`, %d folds, %d epochs with the validation fold scored after every "
      "epoch. One backbone, full label set including N/A, so these line up with "
      "the `dedicated, 1 backbone` rows of script 15."
      % (BACKBONE, args.folds, args.epochs))
    A("")

    A("## Validation macro-F1 by epoch, averaged over folds")
    A("")
    hdr = [str(e + 1) for e in range(args.epochs)]
    A("| subtask | %s |" % " | ".join(hdr))
    A("|---|%s" % ("---:|" * args.epochs))
    for f, _ in fields:
        m = curves[f].mean(axis=0)
        cells = []
        for e in range(args.epochs):
            cells.append("**%.3f**" % m[e] if e == int(m.argmax()) else "%.3f" % m[e])
        A("| `%s` | %s |" % (f, " | ".join(cells)))
    A("")
    A("| subtask | epoch 4 | best epoch | best value | gain from 4 to best |")
    A("|---|---:|---:|---:|---:|")
    for f, _ in fields:
        m = curves[f].mean(axis=0)
        b = int(m.argmax())
        A("| `%s` | %.4f | %d | %.4f | **%+.4f** |"
          % (f, m[3] if len(m) > 3 else float("nan"), b + 1, m[b],
             m[b] - (m[3] if len(m) > 3 else np.nan)))
    A("")

    A("## Official score at a fixed budget vs best-epoch selection")
    A("")
    A("Best epoch is chosen per subtask on out-of-fold validation only; the "
      "test set is scored once per configuration.")
    A("")

    def score(pick):
        tot, comp = 0.0, {}
        for f, wgt in fields:
            p = test_by_ep[f][pick[f]].argmax(1)
            m = f1_score(gold_te[f], p, average="macro", zero_division=0)
            comp[f] = m
            tot += wgt * m
        return tot, comp

    if len(fields) < len(FIELDS):
        A("Only %s were run, so the totals below are partial and comparable "
          "only to each other." % ", ".join("`%s`" % f for f, _ in fields))
        A("")
    rows = [("fixed 4 epochs", {f: 3 for f, _ in FIELDS}),
            ("fixed %d epochs" % args.epochs,
             {f: args.epochs - 1 for f, _ in fields}),
            ("best epoch per subtask (OOF-selected)",
             {f: int(curves[f].mean(axis=0).argmax()) for f, _ in fields})]
    A("| configuration | total | PS (.20) | ES (.30) | EQ (.35) | VT (.15) |")
    A("|---|---:|---:|---:|---:|---:|")
    for nm, pick in rows:
        t, c = score(pick)
        A("| %s | **%.4f** | %.4f | %.4f | %.4f | %.4f |"
          % (nm, t, c["promise_status"], c["evidence_status"],
             c["evidence_quality"], c["verification_timeline"]))
    A("| script 15, dedicated 1 backbone, free | 0.6015 | 0.8111 | 0.6764 | 0.4246 | 0.5848 |")
    A("| competition ensemble, joint, 3 backbones, 15 epochs | 0.6384 | 0.8191 | 0.7132 | 0.4642 | 0.6545 |")
    A("")

    json.dump({f: curves[f].mean(axis=0).tolist() for f, _ in fields},
              open(os.path.join(CACHE, "curves.json"), "w"), indent=1)
    dest = os.path.join(OUT, "training_budget_curve.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
