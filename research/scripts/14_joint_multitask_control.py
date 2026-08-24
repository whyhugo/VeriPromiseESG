"""The control that makes scripts 12 and 13 interpretable.

Script 13 compared a dedicated-head cascade (single encoder) against the
competition submission (3 backbones x 5 folds) and found -0.0324. That number
confounds three things at once:

  1. dedicated heads vs one shared encoder with four heads
  2. a single model vs a three-backbone ensemble
  3. hierarchy enforced by construction vs free prediction

This script removes confounds 2 and 3 by training the joint four-head
multi-task model on the *same* encoder, the *same* folds and the *same* seed as
script 13, with each head predicting its full label set including N/A. Comparing
it against script 13 isolates confound 1, and comparing its free predictions
against its own repaired predictions isolates confound 3.

Three systems, all from the same 5-fold run:

  joint-free      four heads, independent argmax, hierarchy not enforced
  joint-repaired  same probabilities, projected onto the feasible set
  (script 13)     dedicated cascade, hierarchy true by construction

5-fold CV on train+val, test touched once.
"""
import argparse
import csv
import io
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
SUBS = os.path.join(ROOT, "submissions")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824

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


class MultiDS(Dataset):
    def __init__(self, rows, tok, maxlen, with_y=True):
        self.rows, self.tok, self.m, self.with_y = rows, tok, maxlen, with_y

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        e = self.tok(r["data"] or "", truncation=True, max_length=self.m,
                     padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in e.items()}
        ys = []
        for f, _ in FIELDS:
            ys.append(LABELS[f].index(na(r[f])) if self.with_y and na(r[f]) in LABELS[f] else 0)
        return item, torch.tensor(ys)


class JointModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.enc = AutoModel.from_pretrained(name)
        h = AutoConfig.from_pretrained(name).hidden_size
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleList(
            [nn.Linear(h, len(LABELS[f])) for f, _ in FIELDS])

    def forward(self, **enc):
        out = self.enc(**enc).last_hidden_state[:, 0]
        out = self.drop(out)
        return [h(out) for h in self.heads]


def run_fold(tr_rows, eval_rows_list, model_name, maxlen, epochs, bs, lr):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = JointModel(model_name).to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(MultiDS(tr_rows, tok, maxlen), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, len(dl) * epochs), pct_start=0.1)

    weights = []
    for j, (f, _) in enumerate(FIELDS):
        y = np.array([LABELS[f].index(na(r[f])) for r in tr_rows
                      if na(r[f]) in LABELS[f]])
        cnt = np.bincount(y, minlength=len(LABELS[f])).astype(float)
        weights.append(torch.tensor(len(y) / (len(LABELS[f]) * np.maximum(cnt, 1)),
                                    dtype=torch.float32, device="cuda"))
    losses = [nn.CrossEntropyLoss(weight=w) for w in weights]
    scaler = torch.amp.GradScaler("cuda")

    mdl.train()
    for _ in range(epochs):
        for enc, ys in dl:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            ys = ys.to("cuda")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outs = mdl(**enc)
                loss = sum(w * losses[j](outs[j].float(), ys[:, j])
                           for j, (_, w) in enumerate(FIELDS))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

    mdl.eval()
    results = []
    for rows in eval_rows_list:
        dle = DataLoader(MultiDS(rows, tok, maxlen, with_y=False), batch_size=bs * 2)
        acc = [[] for _ in FIELDS]
        with torch.no_grad():
            for enc, _ in dle:
                enc = {k: v.to("cuda") for k, v in enc.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outs = mdl(**enc)
                for j, o in enumerate(outs):
                    acc[j].append(torch.softmax(o.float(), -1).cpu().numpy())
        results.append([np.concatenate(a) for a in acc])
    del mdl
    torch.cuda.empty_cache()
    return results


def official(gold_rows, pm):
    tot, comp = 0.0, {}
    for f, w in FIELDS:
        y = [na(r[f]) for r in gold_rows]
        p = [na(pm[r["id"]][f]) for r in gold_rows]
        m = f1_score(y, p, average="macro", zero_division=0)
        comp[f] = m
        tot += w * m
    return tot, comp


def scr(pm, rows):
    bad = 0
    for r in rows:
        p = pm[r["id"]]
        ps, tl = na(p["promise_status"]), na(p["verification_timeline"])
        es, eq = na(p["evidence_status"]), na(p["evidence_quality"])
        v = False
        if ps == "No":
            v = tl != "N/A" or es != "N/A" or eq != "N/A"
        else:
            v = tl == "N/A" or es == "N/A"
        if es == "No" and eq != "N/A":
            v = True
        if es == "Yes" and eq == "N/A":
            v = True
        bad += bool(v)
    return 100.0 * (len(rows) - bad) / len(rows), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    te = load("vpesg4k_test_2000.csv")
    strat = np.array([LABELS["evidence_quality"].index(na(r["evidence_quality"]))
                      if na(r["evidence_quality"]) in LABELS["evidence_quality"] else 0
                      for r in trv])

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    test_prob = [np.zeros((len(te), len(LABELS[f]))) for f, _ in FIELDS]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for k, (ia, _) in enumerate(skf.split(np.zeros(len(trv)), strat)):
        res = run_fold([trv[i] for i in ia], [te], args.model, args.maxlen,
                       args.epochs, args.bs, args.lr)
        for j in range(len(FIELDS)):
            test_prob[j] += res[0][j] / args.folds
        print("fold %d/%d done" % (k + 1, args.folds), flush=True)

    np.savez(os.path.join(CACHE, "joint_multitask_test_probs.npz"),
             te_ids=np.array([r["id"] for r in te]),
             **{f: test_prob[j] for j, (f, _) in enumerate(FIELDS)})

    free = {}
    for i, r in enumerate(te):
        free[r["id"]] = {f: LABELS[f][int(test_prob[j][i].argmax())]
                         for j, (f, _) in enumerate(FIELDS)}

    rep = {}
    for i, r in enumerate(te):
        row = dict(free[r["id"]])
        if row["promise_status"] == "No":
            row.update({"verification_timeline": "N/A", "evidence_status": "N/A",
                        "evidence_quality": "N/A"})
        else:
            if row["verification_timeline"] == "N/A":
                j = [x for x, _ in FIELDS].index("verification_timeline")
                order = np.argsort(-test_prob[j][i])
                row["verification_timeline"] = next(
                    LABELS["verification_timeline"][o] for o in order
                    if LABELS["verification_timeline"][o] != "N/A")
            if row["evidence_status"] == "N/A":
                row["evidence_status"] = "Yes"
        if row["evidence_status"] != "Yes":
            row["evidence_quality"] = "N/A"
        elif row["evidence_quality"] == "N/A":
            j = [x for x, _ in FIELDS].index("evidence_quality")
            order = np.argsort(-test_prob[j][i])
            row["evidence_quality"] = next(
                LABELS["evidence_quality"][o] for o in order
                if LABELS["evidence_quality"][o] != "N/A")
        rep[r["id"]] = row

    L = []
    A = L.append
    A("# Joint multi-task control")
    A("")
    A("Generated by `research/scripts/14_joint_multitask_control.py`. Encoder "
      "`%s`, %d folds, seed %d -- identical to script 13, so the only difference "
      "is the formulation." % (args.model, args.folds, SEED))
    A("")
    A("| system | total | PS (.20) | ES (.30) | EQ (.35) | VT (.15) | SCR |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for tag, pm in [("joint multi-task, free argmax", free),
                    ("joint multi-task, projected onto feasible set", rep)]:
        t, c = official(te, pm)
        s, _ = scr(pm, te)
        A("| %s | **%.4f** | %.4f | %.4f | %.4f | %.4f | %.1f%% |"
          % (tag, t, c["promise_status"], c["evidence_status"],
             c["evidence_quality"], c["verification_timeline"], s))
    A("| dedicated cascade (script 13) | 0.6061 | 0.8134 | 0.6863 | 0.4118 | 0.6225 | 100.0% |")
    A("| competition ensemble, 3 backbones x 5 folds | 0.6384 | 0.8191 | 0.7132 | 0.4642 | 0.6545 | 94.2% |")
    A("")
    A("The first two rows share every model weight, so their difference is the "
      "cost of enforcing the hierarchy alone. Rows one and three share encoder, "
      "folds and seed, so their difference is the cost of the formulation alone. "
      "Only the last row differs in ensembling.")
    A("")

    dest = os.path.join(OUT, "joint_multitask_control.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
