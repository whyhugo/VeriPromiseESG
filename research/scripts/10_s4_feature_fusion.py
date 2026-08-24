"""S4: do the guideline regex features carry information the transformer misses?

Script 08 found that a logistic regression on 14 regex features taken straight
from the annotation guideline scores macro-F1 0.6518 on Clear vs Not Clear,
against 0.6602 for our 3-backbone x 5-fold transformer ensemble. A gap of
0.0084. Either the transformer has already learned exactly those surface cues,
or the two are picking up different things and fusing them will help.

This settles it by stacking rather than by a custom architecture:

  1. fine-tune a Chinese encoder on binary Clear vs Not Clear, 5-fold CV over
     train+val, producing out-of-fold probabilities and fold-averaged test
     probabilities
  2. compare transformer alone / regex alone / stacked
  3. report the complementarity directly -- how often they disagree, and who is
     right when they do

Misleading is excluded here (n=2 train, n=3 test) and handled separately by the
relatedness work in script 09.

Note this conditions on gold evidence_status=Yes, so it isolates the
evidence_quality problem rather than measuring a deployable pipeline. The
end-to-end number comes later.

Out-of-fold and test probabilities are cached so S3 and S5 can reuse them.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824

FEATURES = [
    ("has_digit", r"[0-9０-９]"),
    ("has_percent", r"[%％]|百分之"),
    ("has_year", r"(19|20)\d{2}\s*年|民國\s*\d+\s*年"),
    ("has_date_unit", r"\d+\s*[月日季]|第?\s*[一二三四]\s*季"),
    ("has_quantity_unit", r"(公?噸|kg|公斤|度|kWh|MWh|MW|GJ|噸CO2|tCO2e|件|人次|小時|平方公尺|公升)"),
    ("has_currency", r"(新臺幣|新台幣|NT\$|元|億元|萬元|千元)"),
    ("has_org_unit", r"(委員會|工作小組|專案小組|部門|事業處|中心|辦公室|治理單位|董事會)"),
    ("has_role", r"(負責|主責|由.{0,8}(部|處|中心|委員會|小組).{0,4}(執行|推動|負責))"),
    ("has_target_word", r"(目標|預計|預期|規劃於|將於|達成率|完成率)"),
    ("has_delta", r"(較.{0,6}(年|期)|相[比較]|成長|增加|減少|下降|提升|降低).{0,6}[0-9０-９]"),
    ("has_certification", r"(ISO\s?\d+|認證|查證|第三方|外部稽核|保證|GRI|SASB|TCFD)"),
    ("has_enumeration", r"(第[一二三四五六七八九]|其一|其二|包括|包含|項目如下|分別為)"),
]


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def ok(v):
    return (v or "").strip() not in ("", "N/A")


def binary_rows(rs):
    out = []
    for r in rs:
        q = (r["evidence_quality"] or "").strip()
        if r["evidence_status"] != "Yes" or q not in ("Clear", "Not Clear"):
            continue
        if not ok(r["evidence_string"]):
            continue
        out.append({"id": r["id"], "y": 1 if q == "Clear" else 0,
                    "para": r["data"] or "",
                    "ev": r["evidence_string"].strip(),
                    "pr": r["promise_string"].strip() if ok(r["promise_string"]) else ""})
    return out


def feat_matrix(rows):
    X = np.zeros((len(rows), len(FEATURES) + 2))
    for i, r in enumerate(rows):
        t = r["ev"]
        for j, (_, pat) in enumerate(FEATURES):
            X[i, j] = 1.0 if re.search(pat, t) else 0.0
        X[i, len(FEATURES)] = len(t)
        X[i, len(FEATURES) + 1] = len(re.findall(r"[0-9０-９]", t)) / max(1, len(t)) * 100
    return X


class DS(Dataset):
    def __init__(self, rows, tok, maxlen, field):
        self.rows, self.tok, self.maxlen, self.field = rows, tok, maxlen, field

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r[self.field], truncation=True, max_length=self.maxlen,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(r["y"])


def train_fold(train_rows, eval_sets, model_name, maxlen, epochs, bs, lr, field):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(DS(train_rows, tok, maxlen, field), batch_size=bs, shuffle=True,
                    drop_last=False)
    steps = max(1, len(dl) * epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    n_pos = sum(r["y"] for r in train_rows)
    n_neg = len(train_rows) - n_pos
    w = torch.tensor([len(train_rows) / (2 * max(1, n_neg)),
                      len(train_rows) / (2 * max(1, n_pos))],
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
                out = mdl(**enc).logits
                loss = lossf(out.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

    mdl.eval()
    preds = []
    for rows in eval_sets:
        dle = DataLoader(DS(rows, tok, maxlen, field), batch_size=bs * 2, shuffle=False)
        ps = []
        with torch.no_grad():
            for enc, _ in dle:
                enc = {k: v.to("cuda") for k, v in enc.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logit = mdl(**enc).logits.float()
                ps.append(torch.softmax(logit, -1)[:, 1].cpu().numpy())
        preds.append(np.concatenate(ps) if ps else np.array([]))
    del mdl
    torch.cuda.empty_cache()
    return preds


def macro(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    ap.add_argument("--field", default="para", choices=["para", "ev"])
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tr = binary_rows(load("vpesg4k_train_1000") + load("vpesg4k_val_1000"))
    te = binary_rows(load("vpesg4k_test_2000.csv"))
    ytr = np.array([r["y"] for r in tr])
    yte = np.array([r["y"] for r in te])
    print("train+val %d (Clear %d / Not Clear %d), test %d (Clear %d / Not Clear %d)"
          % (len(tr), ytr.sum(), len(tr) - ytr.sum(), len(te), yte.sum(),
             len(te) - yte.sum()), flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    oof = np.zeros(len(tr))
    test_p = np.zeros(len(te))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for k, (ia, ib) in enumerate(skf.split(np.zeros(len(tr)), ytr)):
        sub = [tr[i] for i in ia]
        val = [tr[i] for i in ib]
        pv, pt = train_fold(sub, [val, te], args.model, args.maxlen,
                            args.epochs, args.bs, args.lr, args.field)
        oof[ib] = pv
        test_p += pt / args.folds
        print("fold %d/%d  val macroF1 %.4f" % (k + 1, args.folds,
                                                macro(ytr[ib], (pv > .5).astype(int))),
              flush=True)

    np.savez(os.path.join(CACHE, "s4_binary_probs.npz"),
             tr_ids=np.array([r["id"] for r in tr]), oof=oof, ytr=ytr,
             te_ids=np.array([r["id"] for r in te]), test_p=test_p, yte=yte)

    Ftr, Fte = feat_matrix(tr), feat_matrix(te)
    L = []
    A = L.append
    A("# S4: transformer vs guideline features vs both")
    A("")
    A("Generated by `research/scripts/10_s4_feature_fusion.py`. Encoder `%s`, "
      "input `%s`, %d-fold CV on train+val, evaluated once on test."
      % (args.model, args.field, args.folds))
    A("")
    A("Binary Clear vs Not Clear, conditioned on gold `evidence_status=Yes`. "
      "train+val n=%d (Clear %d / Not Clear %d), test n=%d (Clear %d / Not Clear %d)."
      % (len(tr), ytr.sum(), len(tr) - ytr.sum(), len(te), yte.sum(),
         len(te) - yte.sum()))
    A("")

    tf_oof_pred = (oof > .5).astype(int)
    tf_te_pred = (test_p > .5).astype(int)

    sc = StandardScaler().fit(Ftr)
    lr_feat = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=SEED).fit(sc.transform(Ftr), ytr)
    feat_te_prob = lr_feat.predict_proba(sc.transform(Fte))[:, 1]
    feat_te_pred = (feat_te_prob > .5).astype(int)

    Str = np.column_stack([oof, Ftr])
    Ste = np.column_stack([test_p, Fte])
    sc2 = StandardScaler().fit(Str)
    stack = LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SEED).fit(sc2.transform(Str), ytr)
    stack_te_prob = stack.predict_proba(sc2.transform(Ste))[:, 1]
    stack_te_pred = (stack_te_prob > .5).astype(int)

    A("| system | CV macro-F1 (OOF) | test macro-F1 | test ROC-AUC |")
    A("|---|---:|---:|---:|")
    A("| transformer alone | %.4f | %.4f | %.4f |"
      % (macro(ytr, tf_oof_pred), macro(yte, tf_te_pred), roc_auc_score(yte, test_p)))
    A("| guideline regex features alone | %.4f | %.4f | %.4f |"
      % (macro(ytr, lr_feat.predict(sc.transform(Ftr))), macro(yte, feat_te_pred),
         roc_auc_score(yte, feat_te_prob)))
    A("| stacked (transformer prob + features) | %.4f | %.4f | %.4f |"
      % (macro(ytr, stack.predict(sc2.transform(Str))), macro(yte, stack_te_pred),
         roc_auc_score(yte, stack_te_prob)))
    A("")
    A("Reference from earlier work on this same test subset: our 3-backbone "
      "x 5-fold competition ensemble scores macro-F1 0.6602, the regex-only "
      "model 0.6518, always-predict-Clear 0.4537.")
    A("")

    A("## Complementarity")
    A("")
    dis = tf_te_pred != feat_te_pred
    both_right = ((tf_te_pred == yte) & (feat_te_pred == yte)).sum()
    tf_only = ((tf_te_pred == yte) & (feat_te_pred != yte)).sum()
    ft_only = ((tf_te_pred != yte) & (feat_te_pred == yte)).sum()
    neither = ((tf_te_pred != yte) & (feat_te_pred != yte)).sum()
    A("| | count | share |")
    A("|---|---:|---:|")
    for nm, v in [("both right", both_right), ("transformer only right", tf_only),
                  ("features only right", ft_only), ("neither right", neither)]:
        A("| %s | %d | %.1f%% |" % (nm, v, 100.0 * v / len(yte)))
    A("")
    A("They disagree on %d of %d test rows (%.1f%%). Pearson correlation between "
      "the two probability vectors: %.4f."
      % (dis.sum(), len(yte), 100.0 * dis.mean(),
         float(np.corrcoef(test_p, feat_te_prob)[0, 1])))
    A("")
    A("If `features only right` is a meaningful share, the regex cues carry "
      "signal the encoder does not, and the stack should beat both. If it is "
      "near zero, the encoder has already absorbed them and S4 is a dead end.")
    A("")

    dest = os.path.join(OUT, "s4_feature_fusion.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
