"""Does the dedicated-head result generalise to all four subtasks?

S4 showed a single dedicated head scoring 0.7104 on Clear vs Not Clear against
0.6602 for the whole 3-backbone joint multi-task ensemble. If folding a subtask
into the shared four-head setup costs that much, the same should show up on the
other weak subtask, verification_timeline, which the competition system leaves at
0.6545 and the official LLM+RAG baseline at 0.4227.

This trains one dedicated classifier per subtask, each on the conditioning its
label definition implies, then composes them into a full submission with the
annotation hierarchy enforced by construction:

  promise_status        binary, all rows
  evidence_status       binary, trained on gold promise=Yes
  verification_timeline 4-way,  trained on gold promise=Yes
  evidence_quality      binary, trained on gold evidence=Yes

At inference the cascade uses *predicted* upstream labels, so promise=No forces
the other three to N/A and evidence!=Yes forces quality to N/A. Structural
consistency is therefore 100% by construction rather than by post-hoc repair,
which script 05 measured as costing 0.003-0.005.

Misleading is never predicted (2 training instances).

5-fold CV on train+val for out-of-fold estimates, test touched once.
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

TASKS = {
    "promise_status": {"labels": ["No", "Yes"], "cond": None},
    "evidence_status": {"labels": ["No", "Yes"], "cond": ("promise_status", "Yes")},
    "verification_timeline": {
        "labels": ["already", "within_2_years", "between_2_and_5_years",
                   "more_than_5_years"],
        "cond": ("promise_status", "Yes")},
    "evidence_quality": {"labels": ["Not Clear", "Clear"],
                         "cond": ("evidence_status", "Yes")},
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


def run_fold(tr_txt, tr_y, eval_txts, nlab, model_name, maxlen, epochs, bs, lr):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=nlab).to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(DS(tr_txt, tr_y, tok, maxlen), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, len(dl) * epochs), pct_start=0.1)
    cnt = np.bincount(tr_y, minlength=nlab).astype(float)
    w = torch.tensor(len(tr_y) / (nlab * np.maximum(cnt, 1)),
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
                ps.append(torch.softmax(lg, -1).cpu().numpy())
        outs.append(np.concatenate(ps) if ps else np.zeros((0, nlab)))
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
    te_txt = [r["data"] or "" for r in te]
    base = {r["id"]: dict(r) for r in
            csv.DictReader(open(os.path.join(SUBS, args.base_sub), encoding="utf-8-sig"))}

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    test_prob = {}
    oof_report = {}

    for task, spec in TASKS.items():
        labels = spec["labels"]
        pool = []
        for r in trv:
            if spec["cond"]:
                cf, cv = spec["cond"]
                if na(r[cf]) != cv:
                    continue
            v = na(r[task])
            if v in labels:
                pool.append(r)
        txt = [r["data"] or "" for r in pool]
        y = np.array([labels.index(na(r[task])) for r in pool])
        print("[%s] pool %d  dist %s" % (task, len(pool),
                                         dict(zip(labels, np.bincount(y, minlength=len(labels))))),
              flush=True)

        oof = np.zeros((len(pool), len(labels)))
        tp = np.zeros((len(te), len(labels)))
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        for k, (ia, ib) in enumerate(skf.split(np.zeros(len(pool)), y)):
            pv, pt = run_fold([txt[i] for i in ia], y[ia],
                              [[txt[i] for i in ib], te_txt], len(labels),
                              args.model, args.maxlen, args.epochs, args.bs, args.lr)
            oof[ib] = pv
            tp += pt / args.folds
            print("  [%s] fold %d/%d" % (task, k + 1, args.folds), flush=True)
        test_prob[task] = tp
        oof_f1 = f1_score(y, oof.argmax(1), average="macro", zero_division=0)
        oof_report[task] = oof_f1
        print("[%s] OOF macro-F1 %.4f" % (task, oof_f1), flush=True)

    np.savez(os.path.join(CACHE, "dedicated_heads_test_probs.npz"),
             te_ids=np.array([r["id"] for r in te]),
             **{k: v for k, v in test_prob.items()})

    # ---- compose with the hierarchy enforced by construction ----
    pm = {}
    for i, r in enumerate(te):
        rid = r["id"]
        ps = TASKS["promise_status"]["labels"][int(test_prob["promise_status"][i].argmax())]
        row = {"id": rid, "promise_status": ps}
        if ps == "No":
            row.update({"verification_timeline": "N/A", "evidence_status": "N/A",
                        "evidence_quality": "N/A"})
        else:
            row["verification_timeline"] = TASKS["verification_timeline"]["labels"][
                int(test_prob["verification_timeline"][i].argmax())]
            es = TASKS["evidence_status"]["labels"][
                int(test_prob["evidence_status"][i].argmax())]
            row["evidence_status"] = es
            row["evidence_quality"] = (
                TASKS["evidence_quality"]["labels"][
                    int(test_prob["evidence_quality"][i].argmax())]
                if es == "Yes" else "N/A")
        pm[rid] = row

    with open(os.path.join(SUBS, "dedicated_heads.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + [k for k, _ in FIELDS])
        w.writeheader()
        for r in te:
            w.writerow({k: pm[r["id"]].get(k, "") for k in ["id"] + [k for k, _ in FIELDS]})

    L = []
    A = L.append
    A("# Dedicated heads for all four subtasks")
    A("")
    A("Generated by `research/scripts/13_dedicated_heads_full.py`. Encoder `%s`, "
      "%d-fold CV on train+val, test touched once. Each head is trained on the "
      "conditioning its label definition implies; the cascade at inference uses "
      "predicted upstream labels, so the annotation hierarchy holds by "
      "construction." % (args.model, args.folds))
    A("")
    A("| subtask | OOF macro-F1 |")
    A("|---|---:|")
    for t, v in oof_report.items():
        A("| `%s` | %.4f |" % (t, v))
    A("")

    tot0, c0 = official(te, base)
    tot1, c1 = official(te, pm)
    A("| system | total | PS (.20) | ES (.30) | EQ (.35) | VT (.15) |")
    A("|---|---:|---:|---:|---:|---:|")
    A("| official LLM w/o RAG | 0.4956 | 0.7298 | 0.5969 | 0.3549 | 0.3090 |")
    A("| official LLM w/ RAG | 0.5111 | 0.7152 | 0.5958 | 0.3599 | 0.4227 |")
    A("| competition ensemble `%s` | %.4f | %.4f | %.4f | %.4f | %.4f |"
      % (args.base_sub, tot0, c0["promise_status"], c0["evidence_status"],
         c0["evidence_quality"], c0["verification_timeline"]))
    A("| **dedicated heads (this work)** | **%.4f** | %.4f | %.4f | %.4f | %.4f |"
      % (tot1, c1["promise_status"], c1["evidence_status"],
         c1["evidence_quality"], c1["verification_timeline"]))
    A("| delta | **%+.4f** | %+.4f | %+.4f | %+.4f | %+.4f |"
      % (tot1 - tot0, c1["promise_status"] - c0["promise_status"],
         c1["evidence_status"] - c0["evidence_status"],
         c1["evidence_quality"] - c0["evidence_quality"],
         c1["verification_timeline"] - c0["verification_timeline"]))
    A("")

    bad = 0
    for r in te:
        p = pm[r["id"]]
        if p["promise_status"] == "No":
            if any(p[k] != "N/A" for k in ["verification_timeline", "evidence_status",
                                           "evidence_quality"]):
                bad += 1
        elif p["evidence_status"] != "Yes" and p["evidence_quality"] != "N/A":
            bad += 1
    A("Structural consistency rate: **%.1f%%** (%d violations of %d), against "
      "94.2%% for the competition ensemble."
      % (100.0 * (len(te) - bad) / len(te), bad, len(te)))
    A("")

    dest = os.path.join(OUT, "dedicated_heads_full.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
