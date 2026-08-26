"""The combination the decomposition in script 14 pointed at but never ran.

Script 14 separated three effects, each measured against a matched control:

    enforcing the label hierarchy    -0.0130   (same weights, projected vs free)
    dedicated instead of joint heads +0.0176   (both at 100% consistency)
    ensembling three backbones       ~+0.03    (one backbone vs three)

If they are additive, dedicated + three backbones + no enforcement should land
near 0.649 against 0.6384 for the competition system. Additivity was an
assumption, so this script runs the combination instead of extrapolating it.

The design difference from script 13: there the four heads were *conditioned*,
each trained only on rows its label definition applies to, and composed as a
cascade over predicted upstream labels. That makes consistency unavoidable and
therefore cannot test the third arm. Here each subtask gets its own model
trained on all 2,000 rows over its *full* label set including N/A, so every
model decides N/A for itself and the four predictions are free to disagree.

Producing all four cells of the ensembling x consistency grid, with script 14
supplying the joint-formulation rows, gives a 2x2x2 over formulation,
ensembling and consistency.

Per (subtask, backbone) probabilities are cached so the run is resumable.
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
SUBS = os.path.join(ROOT, "submissions")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s15")
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
BACKBONES = ["hfl/chinese-roberta-wwm-ext-large",
             "hfl/chinese-macbert-large",
             "IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese"]


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
        outs.append(np.concatenate(ps))
    del mdl
    torch.cuda.empty_cache()
    return outs


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
    return 100.0 * (len(rows) - bad) / len(rows)


def compose_free(te, probs):
    out = {}
    for i, r in enumerate(te):
        out[r["id"]] = {f: LABELS[f][int(probs[f][i].argmax())] for f, _ in FIELDS}
    return out


def compose_projected(te, probs):
    """Project onto the feasible set, preferring the model's own ranking."""

    def best_not_na(f, i):
        order = np.argsort(-probs[f][i])
        return next(LABELS[f][o] for o in order if LABELS[f][o] != "N/A")

    out = {}
    for i, r in enumerate(te):
        row = {f: LABELS[f][int(probs[f][i].argmax())] for f, _ in FIELDS}
        if row["promise_status"] == "No":
            row["verification_timeline"] = "N/A"
            row["evidence_status"] = "N/A"
            row["evidence_quality"] = "N/A"
        else:
            if row["verification_timeline"] == "N/A":
                row["verification_timeline"] = best_not_na("verification_timeline", i)
            if row["evidence_status"] == "N/A":
                row["evidence_status"] = best_not_na("evidence_status", i)
        if row["evidence_status"] != "Yes":
            row["evidence_quality"] = "N/A"
        elif row["evidence_quality"] == "N/A":
            row["evidence_quality"] = best_not_na("evidence_quality", i)
        out[r["id"]] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    te = load("vpesg4k_test_2000.csv")
    te_txt = [r["data"] or "" for r in te]
    base = {r["id"]: dict(r) for r in
            csv.DictReader(open(os.path.join(SUBS, "ens_3way_tuned.csv"),
                                encoding="utf-8-sig"))}

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    oof_scores = {}
    per_backbone = {}
    for bi, bb in enumerate(BACKBONES):
        per_backbone[bb] = {}
        for f, _ in FIELDS:
            tag = "%d_%s" % (bi, f.replace("_", ""))
            cf = os.path.join(CACHE, "%s.npz" % tag)
            if os.path.exists(cf):
                z = np.load(cf)
                per_backbone[bb][f] = z["test_p"]
                oof_scores[(bb, f)] = float(z["oof_f1"])
                print("[cached] %s %s  OOF %.4f" % (bb, f, z["oof_f1"]), flush=True)
                continue

            labels = LABELS[f]
            pool = [r for r in trv if na(r[f]) in labels]
            txt = [r["data"] or "" for r in pool]
            y = np.array([labels.index(na(r[f])) for r in pool])
            oof = np.zeros((len(pool), len(labels)))
            tp = np.zeros((len(te), len(labels)))
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                  random_state=SEED)
            for k, (ia, ib) in enumerate(skf.split(np.zeros(len(pool)), y)):
                pv, pt = run_fold([txt[i] for i in ia], y[ia],
                                  [[txt[i] for i in ib], te_txt], len(labels),
                                  bb, args.maxlen, args.epochs, args.bs, args.lr)
                oof[ib] = pv
                tp += pt / args.folds
                print("  [%s | %s] fold %d/%d" % (bb.split("/")[-1], f, k + 1,
                                                  args.folds), flush=True)
            of1 = f1_score(y, oof.argmax(1), average="macro", zero_division=0)
            np.savez(cf, test_p=tp, oof_f1=of1)
            per_backbone[bb][f] = tp
            oof_scores[(bb, f)] = float(of1)
            print("[%s | %s] OOF macro-F1 %.4f" % (bb.split("/")[-1], f, of1),
                  flush=True)

    ens = {f: np.mean([per_backbone[bb][f] for bb in BACKBONES], axis=0)
           for f, _ in FIELDS}
    single = {f: per_backbone[BACKBONES[0]][f] for f, _ in FIELDS}

    systems = {
        "dedicated, 1 backbone, free": compose_free(te, single),
        "dedicated, 1 backbone, projected": compose_projected(te, single),
        "dedicated, 3 backbones, free": compose_free(te, ens),
        "dedicated, 3 backbones, projected": compose_projected(te, ens),
    }

    with open(os.path.join(SUBS, "dedicated_ens_free.csv"), "w",
              encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id"] + [k for k, _ in FIELDS])
        w.writeheader()
        pm = systems["dedicated, 3 backbones, free"]
        for r in te:
            row = {"id": r["id"]}
            row.update(pm[r["id"]])
            w.writerow(row)

    L = []
    A = L.append
    A("# Dedicated models, three backbones, hierarchy not enforced")
    A("")
    A("Generated by `research/scripts/15_dedicated_ensemble_free.py`. Each "
      "subtask gets its own model trained on all 2,000 train+val rows over its "
      "full label set including N/A, %d folds, seed %d. Backbones: %s."
      % (args.folds, SEED, ", ".join("`%s`" % b for b in BACKBONES)))
    A("")
    A("## Out-of-fold macro-F1 per backbone and subtask")
    A("")
    A("| subtask | %s |" % " | ".join(b.split("/")[-1] for b in BACKBONES))
    A("|---|%s" % ("---:|" * len(BACKBONES)))
    for f, _ in FIELDS:
        A("| `%s` | %s |" % (f, " | ".join("%.4f" % oof_scores[(b, f)]
                                           for b in BACKBONES)))
    A("")
    A("## Official test score")
    A("")
    A("| system | total | PS (.20) | ES (.30) | EQ (.35) | VT (.15) | SCR |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    t0, c0 = official(te, base)
    A("| competition ensemble (joint, 3 backbones) | %.4f | %.4f | %.4f | %.4f | %.4f | 94.2%% |"
      % (t0, c0["promise_status"], c0["evidence_status"], c0["evidence_quality"],
         c0["verification_timeline"]))
    A("| joint, 1 backbone, free (script 14) | 0.6015 | 0.8078 | 0.6801 | 0.4472 | 0.5291 | 86.0% |")
    A("| joint, 1 backbone, projected (script 14) | 0.5885 | 0.8078 | 0.6822 | 0.4091 | 0.5273 | 100.0% |")
    for tag, pm in systems.items():
        t, c = official(te, pm)
        A("| %s | **%.4f** | %.4f | %.4f | %.4f | %.4f | %.1f%% |"
          % (tag, t, c["promise_status"], c["evidence_status"],
             c["evidence_quality"], c["verification_timeline"], scr(pm, te)))
    A("")
    A("## Additivity check")
    A("")
    tf1, _ = official(te, systems["dedicated, 1 backbone, free"])
    tp1, _ = official(te, systems["dedicated, 1 backbone, projected"])
    tf3, _ = official(te, systems["dedicated, 3 backbones, free"])
    tp3, _ = official(te, systems["dedicated, 3 backbones, projected"])
    A("| effect | measured at 1 backbone | measured at 3 backbones |")
    A("|---|---:|---:|")
    A("| enforcing the hierarchy | %+.4f | %+.4f |" % (tp1 - tf1, tp3 - tf3))
    A("| ensembling (1 -> 3 backbones) | %+.4f (free) | %+.4f (projected) |"
      % (tf3 - tf1, tp3 - tp1))
    A("")
    A("Script 14 measured the hierarchy cost at -0.0130 on the joint "
      "formulation with one backbone. If the effect is stable across "
      "formulation and ensembling, the two columns above should agree with it "
      "and with each other.")
    A("")
    A("Extrapolated target from script 14 was 0.649 for the "
      "`dedicated, 3 backbones, free` cell. Measured: **%.4f**." % tf3)
    A("")

    dest = os.path.join(OUT, "dedicated_ensemble_free.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"%s|%s" % (b, f): v for (b, f), v in oof_scores.items()},
              open(os.path.join(CACHE, "oof_scores.json"), "w"), indent=1)
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
