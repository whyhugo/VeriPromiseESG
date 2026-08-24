"""End-to-end: what is the dedicated evidence_quality head worth on the official metric?

S4 found that a single dedicated binary head scores 0.7104 macro-F1 on Clear vs
Not Clear, against 0.6602 for the whole 3-backbone competition ensemble. But
that was conditioned on *gold* evidence_status=Yes, which a deployed system does
not have. This script measures the real gain.

Pipeline: keep promise_status, evidence_status and verification_timeline from the
existing competition submission, and replace evidence_quality with

    predicted evidence_status != Yes  ->  N/A
    otherwise                         ->  dedicated binary head (Clear / Not Clear)

Misleading is never predicted: 2 training instances make it unlearnable, and the
zero-shot routes are covered separately in scripts 09 and 11.

Official metric: 0.20*PS + 0.30*ES + 0.35*EQ + 0.15*VT, all macro-F1.

The binary threshold is chosen on out-of-fold predictions only, never on test.
"""
import argparse
import csv
import io
import os
import re
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
SUBS = os.path.join(ROOT, "submissions")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824
FIELDS = [("promise_status", 0.20), ("evidence_status", 0.30),
          ("evidence_quality", 0.35), ("verification_timeline", 0.15)]


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


def run_fold(tr_txt, tr_y, eval_txts, model_name, maxlen, epochs, bs, lr):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(DS(tr_txt, tr_y, tok, maxlen), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, len(dl) * epochs), pct_start=0.1)
    npos = int(np.sum(tr_y))
    nneg = len(tr_y) - npos
    w = torch.tensor([len(tr_y) / (2 * max(1, nneg)), len(tr_y) / (2 * max(1, npos))],
                     dtype=torch.float32, device="cuda")
    lossf = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda")
    mdl.train()
    for _ in range(epochs):
        for enc, y in dl:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            y = y.to("cuda")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(mdl(**enc).logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
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
                ps.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        outs.append(np.concatenate(ps))
    del mdl
    torch.cuda.empty_cache()
    return outs


def official(gold_rows, pred_map):
    tot, comp = 0.0, {}
    for f, w in FIELDS:
        y = [na(r[f]) for r in gold_rows]
        p = [na(pred_map[r["id"]][f]) for r in gold_rows]
        m = f1_score(y, p, average="macro", zero_division=0)
        comp[f] = m
        tot += w * m
    return tot, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    ap.add_argument("--base-sub", default="ens_3way_tuned.csv")
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    te = load("vpesg4k_test_2000.csv")
    base = {r["id"]: dict(r) for r in
            csv.DictReader(open(os.path.join(SUBS, args.base_sub), encoding="utf-8-sig"))}

    # training pool: gold evidence=Yes with a Clear / Not Clear label
    pool = [r for r in trv if r["evidence_status"] == "Yes"
            and (r["evidence_quality"] or "").strip() in ("Clear", "Not Clear")]
    tr_txt = [r["data"] or "" for r in pool]
    tr_y = np.array([1 if r["evidence_quality"].strip() == "Clear" else 0 for r in pool])
    te_txt = [r["data"] or "" for r in te]
    print("train pool %d (Clear %d / Not Clear %d), test rows %d"
          % (len(pool), tr_y.sum(), len(pool) - tr_y.sum(), len(te)), flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    oof = np.zeros(len(pool))
    test_p = np.zeros(len(te))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for k, (ia, ib) in enumerate(skf.split(np.zeros(len(pool)), tr_y)):
        pv, pt = run_fold([tr_txt[i] for i in ia], tr_y[ia],
                          [[tr_txt[i] for i in ib], te_txt],
                          args.model, args.maxlen, args.epochs, args.bs, args.lr)
        oof[ib] = pv
        test_p += pt / args.folds
        print("fold %d/%d done" % (k + 1, args.folds), flush=True)

    np.savez(os.path.join(CACHE, "e2e_quality_probs.npz"),
             pool_ids=np.array([r["id"] for r in pool]), oof=oof, tr_y=tr_y,
             te_ids=np.array([r["id"] for r in te]), test_p=test_p)

    # threshold picked on OOF only
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(tr_y, (oof > t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    print("OOF best threshold %.2f (macroF1 %.4f)" % (best_t, best_f), flush=True)

    L = []
    A = L.append
    A("# End-to-end gain from a dedicated evidence_quality head")
    A("")
    A("Generated by `research/scripts/12_end_to_end_gain.py`. Encoder `%s`, "
      "%d-fold CV on train+val, threshold %.2f chosen on out-of-fold predictions "
      "only. Base submission `%s` supplies the other three fields unchanged."
      % (args.model, args.folds, best_t, args.base_sub))
    A("")
    A("`evidence_quality` is recomposed as: N/A when the predicted "
      "`evidence_status` is not Yes, otherwise the binary head's Clear / Not "
      "Clear call. Misleading is never predicted.")
    A("")

    variants = {}
    for tag, thr in [("threshold 0.50", 0.5), ("threshold %.2f (OOF-tuned)" % best_t, best_t)]:
        pm = {}
        for i, r in enumerate(te):
            rid = r["id"]
            row = dict(base[rid])
            if na(row["evidence_status"]) != "Yes":
                row["evidence_quality"] = "N/A"
            else:
                row["evidence_quality"] = "Clear" if test_p[i] > thr else "Not Clear"
            pm[rid] = row
        variants[tag] = pm

    A("| system | total | PS (.20) | ES (.30) | EQ (.35) | VT (.15) |")
    A("|---|---:|---:|---:|---:|---:|")
    tot0, c0 = official(te, base)
    A("| baseline `%s` | %.4f | %.4f | %.4f | %.4f | %.4f |"
      % (args.base_sub, tot0, c0["promise_status"], c0["evidence_status"],
         c0["evidence_quality"], c0["verification_timeline"]))
    for tag, pm in variants.items():
        t1, c1 = official(te, pm)
        A("| + dedicated EQ head, %s | **%.4f** | %.4f | %.4f | **%.4f** | %.4f |"
          % (tag, t1, c1["promise_status"], c1["evidence_status"],
             c1["evidence_quality"], c1["verification_timeline"]))
    A("")
    A("Official reference baselines recomputed with the same weights: "
      "LLM w/o RAG 0.4956, LLM w/ RAG 0.5111 (Day et al., Table IV).")
    A("")

    A("## evidence_quality per class")
    A("")
    A("| class | gold n | %s |" % " | ".join("%s F1" % t for t in variants))
    A("|---|---:|%s" % ("---:|" * len(variants)))
    gy = [na(r["evidence_quality"]) for r in te]
    for c in ["Clear", "Not Clear", "Misleading", "N/A"]:
        cells = []
        for tag, pm in variants.items():
            p = [na(pm[r["id"]]["evidence_quality"]) for r in te]
            cells.append("%.4f" % f1_score(gy, p, labels=[c], average="macro",
                                           zero_division=0))
        cells0 = f1_score(gy, [na(base[r["id"]]["evidence_quality"]) for r in te],
                          labels=[c], average="macro", zero_division=0)
        A("| `%s` | %d | %s | (baseline %.4f) |"
          % (c, gy.count(c), " | ".join(cells), cells0))
    A("")

    dest = os.path.join(OUT, "end_to_end_gain.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
