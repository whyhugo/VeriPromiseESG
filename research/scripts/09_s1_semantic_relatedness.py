"""S1 (semantic): does promise-evidence relatedness predict evidence quality?

The annotation guideline defines Misleading relationally -- evidence that "has
no clear connection with the claim and appears to divert attention" -- so a
relatedness score between promise_string and evidence_string is a zero-shot
signal that needs no Misleading training examples. This is the H5 test.

The lexical version (script 08) was negative: char TF-IDF cosine gave AUC 0.5239
on Clear vs Not Clear, i.e. chance. Two things survived, and this script checks
whether a dense model changes either:

  * Not Clear evidence had *higher* lexical overlap with the promise than Clear
    (0.1049 vs 0.0747), consistent with superficial evidence restating the
    promise instead of adding content.
  * length(evidence)/length(promise) did separate the classes (AUC 0.6535).

Scores computed here:
  bge_cos      bi-encoder cosine, BAAI/bge-m3
  rerank       cross-encoder relevance, BAAI/bge-reranker-v2-m3
  nli_entail   optional, if an NLI checkpoint is available

Then: does relatedness add anything on top of the 14 guideline regex features
from script 08? Fit on train+val, evaluated once on test.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824
CLASSES3 = ["Clear", "Not Clear", "Misleading"]

# same 12 regex features as script 08, kept in sync deliberately
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


def rows_with_quality(rs, split):
    out = []
    for r in rs:
        q = (r["evidence_quality"] or "").strip()
        if r["evidence_status"] != "Yes" or q not in CLASSES3:
            continue
        if not (ok(r["evidence_string"]) and ok(r["promise_string"])):
            continue
        out.append({"id": r["id"], "split": split, "y": q,
                    "pr": r["promise_string"].strip(),
                    "ev": r["evidence_string"].strip(),
                    "para": r["data"] or ""})
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


def bi_encoder_cos(rows, model_name, batch=64):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device="cuda")
    pr = m.encode([r["pr"] for r in rows], batch_size=batch, normalize_embeddings=True,
                  show_progress_bar=False, convert_to_numpy=True)
    ev = m.encode([r["ev"] for r in rows], batch_size=batch, normalize_embeddings=True,
                  show_progress_bar=False, convert_to_numpy=True)
    del m
    torch.cuda.empty_cache()
    return (pr * ev).sum(axis=1)


def cross_encoder_score(rows, model_name, batch=32, maxlen=512):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name, dtype=torch.float16).to("cuda").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            enc = tok([r["pr"] for r in chunk], [r["ev"] for r in chunk],
                      padding=True, truncation=True, max_length=maxlen,
                      return_tensors="pt").to("cuda")
            logits = mdl(**enc).logits.float().view(-1)
            out.append(logits.cpu().numpy())
    del mdl
    torch.cuda.empty_cache()
    return np.concatenate(out)


def auc_clear(scores, ys):
    idx = [i for i, y in enumerate(ys) if y in ("Clear", "Not Clear")]
    yy = [1 if ys[i] == "Clear" else 0 for i in idx]
    if len(set(yy)) < 2:
        return float("nan")
    return roc_auc_score(yy, np.asarray(scores)[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi", default="BAAI/bge-m3")
    ap.add_argument("--cross", default="BAAI/bge-reranker-v2-m3")
    args = ap.parse_args()

    tr = rows_with_quality(load("vpesg4k_train_1000"), "train") + \
        rows_with_quality(load("vpesg4k_val_1000"), "val")
    te = rows_with_quality(load("vpesg4k_test_2000.csv"), "test")
    allrows = tr + te
    print("train+val %d, test %d" % (len(tr), len(te)), flush=True)

    scores = {}
    print("encoding with", args.bi, flush=True)
    scores["bge_cos"] = bi_encoder_cos(allrows, args.bi)
    print("scoring with", args.cross, flush=True)
    scores["rerank"] = cross_encoder_score(allrows, args.cross)
    scores["len_ratio"] = np.array([len(r["ev"]) / max(1, len(r["pr"])) for r in allrows])

    np.savez(os.path.join(CACHE, "s1_semantic_scores.npz"),
             ids=np.array([r["id"] for r in allrows]),
             split=np.array([r["split"] for r in allrows]),
             y=np.array([r["y"] for r in allrows]), **scores)

    ys = [r["y"] for r in allrows]
    sp = [r["split"] for r in allrows]
    L, A = [], None
    L = []
    A = L.append
    A("# S1 (semantic): promise-evidence relatedness")
    A("")
    A("Generated by `research/scripts/09_s1_semantic_relatedness.py` on the pod. "
      "Bi-encoder `%s`, cross-encoder `%s`." % (args.bi, args.cross))
    A("")
    A("n = %d (train+val %d, test %d), rows with `evidence_status=Yes` and both "
      "gold spans present." % (len(allrows), len(tr), len(te)))
    A("")
    A("## Class means and separation")
    A("")
    A("| score | Clear | Not Clear | Misleading | AUC Clear-vs-NotClear (all) | AUC (test only) |")
    A("|---|---:|---:|---:|---:|---:|")
    for k in ["bge_cos", "rerank", "len_ratio"]:
        s = np.asarray(scores[k])
        m = {c: s[[i for i, y in enumerate(ys) if y == c]] for c in CLASSES3}
        ti = [i for i, x in enumerate(sp) if x == "test"]
        a_all = auc_clear(s, ys)
        a_te = auc_clear(s[ti], [ys[i] for i in ti])
        A("| `%s` | %.4f | %.4f | %s | %.4f | %.4f |"
          % (k, m["Clear"].mean(), m["Not Clear"].mean(),
             ("%.4f" % m["Misleading"].mean()) if len(m["Misleading"]) else "n/a",
             a_all, a_te))
    A("")
    A("Lexical reference from script 08: char TF-IDF cosine AUC 0.5239, "
      "length ratio AUC 0.6535.")
    A("")

    A("## Where the Misleading instances fall")
    A("")
    mis = [i for i, y in enumerate(ys) if y == "Misleading"]
    A("| id | split | %s |" % " | ".join("`%s` pctile" % k for k in scores))
    A("|---|---|%s" % ("---:|" * len(scores)))
    for i in mis:
        cells = []
        for k in scores:
            s = np.asarray(scores[k])
            cells.append("%.1f" % (100.0 * (s < s[i]).mean()))
        A("| %s | %s | %s |" % (allrows[i]["id"], sp[i], " | ".join(cells)))
    A("")
    A("n=%d, so this is an observation rather than evidence. A useful zero-shot "
      "detector would put all of them near the 0th percentile of relatedness."
      % len(mis))
    A("")

    # ---- does relatedness add on top of the guideline features? ----
    A("## Does relatedness add to the guideline features?")
    A("")
    A("Fit on train+val, evaluated once on test, binary Clear vs Not Clear.")
    A("")
    tr2 = [r for r in tr if r["y"] in ("Clear", "Not Clear")]
    te2 = [r for r in te if r["y"] in ("Clear", "Not Clear")]
    idx_tr = [allrows.index(r) for r in tr2]
    idx_te = [allrows.index(r) for r in te2]
    ya = np.array([1 if r["y"] == "Clear" else 0 for r in tr2])
    yb = np.array([1 if r["y"] == "Clear" else 0 for r in te2])
    Fa, Fb = feat_matrix(tr2), feat_matrix(te2)
    Ra = np.column_stack([np.asarray(scores[k])[idx_tr] for k in scores])
    Rb = np.column_stack([np.asarray(scores[k])[idx_te] for k in scores])

    A("| feature set | macro-F1 | ROC-AUC |")
    A("|---|---:|---:|")
    for name, Xa, Xb in [("guideline regex features only", Fa, Fb),
                         ("relatedness scores only", Ra, Rb),
                         ("both", np.hstack([Fa, Ra]), np.hstack([Fb, Rb]))]:
        sc = StandardScaler().fit(Xa)
        lr = LogisticRegression(max_iter=3000, class_weight="balanced",
                                random_state=SEED).fit(sc.transform(Xa), ya)
        pb = lr.predict(sc.transform(Xb))
        A("| %s | %.4f | %.4f |"
          % (name, f1_score(yb, pb, average="macro", zero_division=0),
             roc_auc_score(yb, lr.predict_proba(sc.transform(Xb))[:, 1])))
    A("")
    A("Reference on the same binary test subset: our transformer ensemble scores "
      "macro-F1 0.6602, the regex-only model 0.6518, always-Clear 0.4537.")
    A("")

    dest = os.path.join(OUT, "s1_semantic_relatedness.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
