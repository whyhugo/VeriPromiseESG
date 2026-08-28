"""The clause tagger: find which part of the paragraph is the promise.

This is the contribution the feasibility study in script 28 pointed at. The
dataset marks, for every paragraph containing a promise, exactly which text the
promise is -- 814 training, 813 validation and 1,625 test rows, at 100% coverage
-- and no participating system used it. Classification says a paragraph contains
a promise; this says which words it is, which is what an auditor actually needs.

Script 28 settled the task design. Annotators marked clauses, not sentences: 92%
of pieces sit inside one sentence but occupy a median 0.41 of it, so selecting
whole sentences caps at character-F1 0.5414. Splitting on the comma as well, 71%
of annotated boundaries land exactly on a clause boundary and the selection
ceiling rises to 0.9748. The task is therefore binary labelling over clauses,
which also handles the fifth of spans made of several disjoint pieces -- they are
simply non-adjacent positive clauses, something a start/end pointer model cannot
represent at all.

Architecture: encode the whole paragraph once so every clause is judged in
context, then mean-pool each clause's tokens and score it with a shared linear
head. Clause boundaries come from the tokeniser's character offsets, so no
alignment heuristics are needed.

Evaluation is the same character-level metric as the baselines, on the same
cleaned gold spans: precision, recall, F1, intersection over union, and exact
match. Five-fold cross-validation over train+val; the test set is scored once.
"""
import argparse
import csv
import io
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s32")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260828
SEP = "｜"
SENT_END = "\u3002\uff01\uff1f\uff1b!?;\n"
CLAUSE_END = SENT_END + "\uff0c\u3001,"
BACKBONE = "hfl/chinese-roberta-wwm-ext-large"
MAXLEN = 384
MAX_CLAUSES = 48

PROMISE_CUES = ["將", "預計", "規劃", "目標", "承諾", "預期", "擬", "持續推動",
                "力求", "期望", "朝", "邁向", "致力", "未來"]


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def ok(v):
    return (v or "").strip() not in ("", "N/A")


def pieces_of(raw):
    out = []
    for p in (raw or "").split(SEP):
        p = re.sub(r"\s*\d+\s*$", "", p.strip()).strip()
        if p:
            out.append(p)
    return out


def clauses_of(para):
    out, buf, start = [], [], 0
    for i, ch in enumerate(para):
        buf.append(ch)
        if ch in CLAUSE_END:
            if "".join(buf).strip():
                out.append((start, i + 1))
            buf, start = [], i + 1
    if buf and "".join(buf).strip():
        out.append((start, len(para)))
    return out


def gold_mask(para, raw):
    m = np.zeros(len(para), dtype=bool)
    for p in pieces_of(raw):
        i = para.find(p)
        if i >= 0:
            m[i:i + len(p)] = True
    return m


def clause_labels(para, raw):
    """A clause is positive when most of its characters are inside the span."""
    g = gold_mask(para, raw)
    return [int(g[a:b].sum() > 0.5 * (b - a)) for a, b in clauses_of(para)]


def prf(pred, gold):
    tp = int((pred & gold).sum())
    if tp == 0:
        return 0.0, 0.0, 0.0, 0.0
    p = tp / max(1, int(pred.sum()))
    r = tp / max(1, int(gold.sum()))
    return p, r, 2 * p * r / (p + r), tp / max(1, int((pred | gold).sum()))


class ClauseDS(Dataset):
    def __init__(self, rows, tok, field):
        self.rows, self.tok, self.field = rows, tok, field

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        para = r["data"] or ""
        enc = self.tok(para, truncation=True, max_length=MAXLEN,
                       padding="max_length", return_offsets_mapping=True,
                       return_tensors="pt")
        off = enc.pop("offset_mapping")[0].numpy()
        cl = clauses_of(para)[:MAX_CLAUSES]
        lab = clause_labels(para, r.get(self.field))[:MAX_CLAUSES]
        # token membership per clause, from the tokeniser's own character offsets
        sel = np.zeros((MAX_CLAUSES, MAXLEN), dtype=np.float32)
        for c, (a, b) in enumerate(cl):
            hit = [t for t in range(MAXLEN)
                   if off[t][1] > off[t][0] and off[t][0] < b and off[t][1] > a]
            if hit:
                sel[c, hit] = 1.0 / len(hit)
        y = np.full(MAX_CLAUSES, -100, dtype=np.int64)
        y[:len(lab)] = lab
        return ({k: v.squeeze(0) for k, v in enc.items()},
                torch.from_numpy(sel), torch.from_numpy(y), i)


class Tagger(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.enc = AutoModel.from_pretrained(name)
        h = AutoConfig.from_pretrained(name).hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, 2)

    def forward(self, enc, sel):
        hs = self.enc(**enc).last_hidden_state          # B x L x H
        pooled = torch.bmm(sel, hs)                     # B x C x H
        return self.head(self.drop(pooled))             # B x C x 2


def run_fold(tr_rows, eval_sets, field, epochs, bs, lr):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    mdl = Tagger(BACKBONE).float().to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(ClauseDS(tr_rows, tok, field), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, len(dl) * epochs), pct_start=0.1)

    pos = sum(sum(clause_labels(r["data"] or "", r.get(field))) for r in tr_rows)
    tot = sum(len(clauses_of(r["data"] or "")) for r in tr_rows)
    w = torch.tensor([tot / (2 * max(1, tot - pos)), tot / (2 * max(1, pos))],
                     dtype=torch.float32, device="cuda")
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-100)

    mdl.train()
    for _ in range(epochs):
        for enc, sel, y, _ in dl:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            sel, y = sel.to("cuda"), y.to("cuda")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logit = mdl(enc, sel).float()
            loss = lossf(logit.reshape(-1, 2), y.reshape(-1))
            loss.backward()
            opt.step()
            sched.step()

    mdl.eval()
    outs = []
    for rows in eval_sets:
        probs = np.zeros((len(rows), MAX_CLAUSES))
        dle = DataLoader(ClauseDS(rows, tok, field), batch_size=bs * 2)
        with torch.no_grad():
            for enc, sel, _, idx in dle:
                enc = {k: v.to("cuda") for k, v in enc.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = mdl(enc, sel.to("cuda")).float()
                p = torch.softmax(lg, -1)[:, :, 1].cpu().numpy()
                for n, j in enumerate(idx.numpy()):
                    probs[j] = p[n]
        outs.append(probs)
    del mdl
    torch.cuda.empty_cache()
    return outs


def score(rows, probs, field, thr=0.5):
    P, R, F, I, EM = [], [], [], [], 0
    for n, r in enumerate(rows):
        para = r["data"] or ""
        g = gold_mask(para, r.get(field))
        pred = np.zeros(len(para), dtype=bool)
        for c, (a, b) in enumerate(clauses_of(para)[:MAX_CLAUSES]):
            if probs[n][c] > thr:
                pred[a:b] = True
        p, rc, f, i = prf(pred, g)
        P.append(p); R.append(rc); F.append(f); I.append(i)
        EM += int(np.array_equal(pred, g))
    return (float(np.mean(P)), float(np.mean(R)), float(np.mean(F)),
            float(np.mean(I)), 100.0 * EM / len(rows))


def _eval(rows, field, pick):
    P, R, F, I, EM = [], [], [], [], 0
    for r in rows:
        para = r["data"] or ""
        g = gold_mask(para, r.get(field))
        pred = pick(para)
        p, rc, f, i = prf(pred, g)
        P.append(p); R.append(rc); F.append(f); I.append(i)
        EM += int(np.array_equal(pred, g))
    return (float(np.mean(P)), float(np.mean(R)), float(np.mean(F)),
            float(np.mean(I)), 100.0 * EM / len(rows))


def baselines(rows, field):
    """The trivial strategies from script 28, scored on these rows.

    Script 28's strongest heuristic worked at sentence level, so comparing the
    tagger only against a clause-level cue rule would flatter it. Both are
    reported.
    """
    def whole(para):
        return np.ones(len(para), dtype=bool)

    def sent_cue(para):
        pred = np.zeros(len(para), dtype=bool)
        buf, start_i, sents = [], 0, []
        for i, ch in enumerate(para):
            buf.append(ch)
            if ch in SENT_END:
                if "".join(buf).strip():
                    sents.append((start_i, i + 1))
                buf, start_i = [], i + 1
        if buf and "".join(buf).strip():
            sents.append((start_i, len(para)))
        hit = [(a, b) for a, b in sents
               if any(c in para[a:b] for c in PROMISE_CUES)]
        for a, b in (hit or sents[:1]):
            pred[a:b] = True
        return pred

    def clause_cue(para):
        pred = np.zeros(len(para), dtype=bool)
        for a, b in clauses_of(para):
            if any(c in para[a:b] for c in PROMISE_CUES):
                pred[a:b] = True
        return pred

    return [("whole paragraph", _eval(rows, field, whole)),
            ("sentences containing a cue (script 28's best)",
             _eval(rows, field, sent_cue)),
            ("clauses containing a cue", _eval(rows, field, clause_cue))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="promise_string")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    trv = [r for r in load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
           if ok(r.get(args.field))]
    te = [r for r in load("vpesg4k_test_2000.csv") if ok(r.get(args.field))]
    print("train+val %d, test %d" % (len(trv), len(te)), flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    oof = np.zeros((len(trv), MAX_CLAUSES))
    tp = np.zeros((len(te), MAX_CLAUSES))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for k, (ia, ib) in enumerate(kf.split(trv)):
        pv, pt = run_fold([trv[i] for i in ia],
                          [[trv[i] for i in ib], te],
                          args.field, args.epochs, args.bs, args.lr)
        for n, j in enumerate(ib):
            oof[j] = pv[n]
        tp += pt / args.folds
        print("fold %d/%d  OOF char-F1 %.4f"
              % (k + 1, args.folds,
                 score([trv[i] for i in ib], pv, args.field)[2]), flush=True)

    np.savez(os.path.join(CACHE, "%s.npz" % args.field), oof=oof, test=tp)

    # threshold chosen on out-of-fold predictions only
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.05):
        f = score(trv, oof, args.field, t)[2]
        if f > best_f:
            best_f, best_t = f, float(t)
    print("OOF best threshold %.2f (char-F1 %.4f)" % (best_t, best_f), flush=True)

    L = []
    A = L.append
    A("# Clause tagger for `%s`" % args.field)
    A("")
    A("Generated by `research/scripts/32_clause_tagger.py`. Encoder `%s`, the "
      "paragraph encoded once so every clause is judged in context, each "
      "clause's tokens mean-pooled and scored by a shared head. %d-fold "
      "cross-validation on train+val, threshold %.2f chosen out-of-fold, test "
      "scored once." % (BACKBONE, args.folds, best_t))
    A("")
    A("train+val %d rows, test %d rows." % (len(trv), len(te)))
    A("")
    A("| system | precision | recall | char-F1 | IoU | exact match |")
    A("|---|---:|---:|---:|---:|---:|")
    bl = baselines(te, args.field)
    for nm, v in bl:
        A("| %s | %.4f | %.4f | %.4f | %.4f | %.1f%% |" % ((nm,) + v))
    b = max(bl, key=lambda kv: kv[1][2])[1]
    o = score(trv, oof, args.field, best_t)
    A("| **clause tagger**, out-of-fold | %.4f | %.4f | **%.4f** | %.4f | %.1f%% |" % o)
    s = score(te, tp, args.field, best_t)
    A("| **clause tagger**, test | %.4f | %.4f | **%.4f** | %.4f | %.1f%% |" % s)
    A("")
    A("The baselines are re-scored on these exact test rows rather than quoted "
      "from script 28, which measured them on train+val. The ceiling of the "
      "clause-selection reduction is 0.9748.")
    A("")
    A("## Reading")
    A("")
    A("The tagger reaches char-F1 **%.4f** on the test set against **%.4f** for "
      "the strongest baseline on the same rows, closing %.0f%% of the distance "
      "to the 0.9748 ceiling."
      % (s[2], b[2], 100 * (s[2] - b[2]) / max(1e-9, 0.9748 - b[2])))
    A("")
    A("Exact match rises from %.1f%% to %.1f%%, which is the share of paragraphs "
      "where the predicted text is character-for-character the annotated span."
      % (b[4], s[4]))
    A("")

    dest = os.path.join(OUT, "clause_tagger_%s.md" % args.field)
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
