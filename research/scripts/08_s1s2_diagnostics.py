"""S1 + S2: two diagnostics that decide which fixes are worth building.

S2 -- guideline-derived specificity features.
The ML-Promise annotation guideline defines Not Clear as information being
"missing to the extent that what is said becomes only partially intelligible
and/or superficial", and states that quality "refers to specificity and detail
-- such as the inclusion of timelines, figures, percentages, identification of
involved parties, or clear descriptions of the intended outcome".

So the features tested here are enumerated by the guideline, not invented. If
they predict Clear vs Not Clear well, the task is largely surface-feature
detection and auxiliary supervision (S4) is well founded. If they do not, the
task needs relational or semantic modelling instead.

S1 -- promise/evidence relatedness.
The guideline defines Misleading relationally: evidence that "has no clear
connection with the claim and appears to divert attention". That makes a
relatedness score between promise_string and evidence_string a zero-shot signal
requiring no Misleading training examples.

Misleading has only 5 instances in the whole corpus, so S1 is evaluated
primarily on Clear vs Not Clear (n=1344 / 226 in train+val terms), where the
result can actually be trusted. Misleading is reported as an observation only.

This run uses lexical relatedness (character n-gram TF-IDF cosine, token
overlap). A dense/NLI version needs a GPU and is the follow-up.

Protocol: all fitting and inspection on train+val. Test is touched once, at the
end, purely to confirm the correlations hold out of sample.
"""
import csv
import io
import os
import re
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
os.makedirs(OUT, exist_ok=True)
SEED = 20260824

# ---- S2 features, each traceable to a phrase in the annotation guideline ----
FEATURES = [
    ("has_digit", r"[0-9０-９]", "figures"),
    ("has_percent", r"[%％]|百分之", "percentages"),
    ("has_year", r"(19|20)\d{2}\s*年|民國\s*\d+\s*年", "timelines"),
    ("has_date_unit", r"\d+\s*[月日季]|第?\s*[一二三四]\s*季", "timelines"),
    ("has_quantity_unit", r"(公?噸|kg|公斤|度|kWh|MWh|MW|GJ|噸CO2|tCO2e|件|人次|小時|平方公尺|公升)", "figures"),
    ("has_currency", r"(新臺幣|新台幣|NT\$|元|億元|萬元|千元)", "figures"),
    ("has_org_unit", r"(委員會|工作小組|專案小組|部門|事業處|中心|辦公室|治理單位|董事會)", "involved parties"),
    ("has_role", r"(負責|主責|由.{0,8}(部|處|中心|委員會|小組).{0,4}(執行|推動|負責))", "involved parties"),
    ("has_target_word", r"(目標|預計|預期|規劃於|將於|達成率|完成率)", "intended outcome"),
    ("has_delta", r"(較.{0,6}(年|期)|相[比較]|成長|增加|減少|下降|提升|降低).{0,6}[0-9０-９]", "figures"),
    ("has_certification", r"(ISO\s?\d+|認證|查證|第三方|外部稽核|保證|GRI|SASB|TCFD)", "intended outcome"),
    ("has_enumeration", r"(第[一二三四五六七八九]|其一|其二|包括|包含|項目如下|分別為)", "intended outcome"),
]
CLASSES3 = ["Clear", "Not Clear", "Misleading"]


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def ok(v):
    return (v or "").strip() not in ("", "N/A")


def rows_with_quality(rs):
    out = []
    for r in rs:
        q = (r["evidence_quality"] or "").strip()
        if r["evidence_status"] != "Yes" or q not in CLASSES3:
            continue
        if not ok(r["evidence_string"]):
            continue
        out.append({
            "id": r["id"],
            "para": r["data"] or "",
            "ev": r["evidence_string"].strip(),
            "pr": r["promise_string"].strip() if ok(r["promise_string"]) else "",
            "y": q,
        })
    return out


def feat_matrix(rows, field):
    X = np.zeros((len(rows), len(FEATURES) + 2))
    for i, r in enumerate(rows):
        t = r[field]
        for j, (_, pat, _) in enumerate(FEATURES):
            X[i, j] = 1.0 if re.search(pat, t) else 0.0
        X[i, len(FEATURES)] = len(t)
        X[i, len(FEATURES) + 1] = len(re.findall(r"[0-9０-９]", t)) / max(1, len(t)) * 100
    return X


def cramers_v(a, b):
    """a: binary feature array, b: binary label array."""
    tab = np.zeros((2, 2))
    for x, y in zip(a, b):
        tab[int(x), int(y)] += 1
    n = tab.sum()
    if n == 0 or tab.sum(axis=0).min() == 0 or tab.sum(axis=1).min() == 0:
        return 0.0
    exp = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / n
    chi2 = ((tab - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
    return float(np.sqrt(chi2 / n))


def char_tfidf_cos(a_list, b_list):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1,
                          sublinear_tf=True)
    vec.fit(a_list + b_list)
    A = vec.transform(a_list)
    B = vec.transform(b_list)
    num = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    den = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel()) * \
        np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
    return num / np.maximum(den, 1e-9)


def jaccard_char(a, b, n=2):
    sa = {a[i:i + n] for i in range(max(0, len(a) - n + 1))}
    sb = {b[i:i + n] for i in range(max(0, len(b) - n + 1))}
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    tr = rows_with_quality(load("vpesg4k_train_1000") + load("vpesg4k_val_1000"))
    te = rows_with_quality(load("vpesg4k_test_2000.csv"))

    L = []
    A = L.append
    A("# S1 + S2 diagnostics")
    A("")
    A("Generated by `research/scripts/08_s1s2_diagnostics.py`. Analysis on "
      "train+val (n=%d); test (n=%d) touched once at the end for out-of-sample "
      "confirmation only." % (len(tr), len(te)))
    A("")
    A("Label counts -- train+val %s, test %s."
      % (dict(Counter(r["y"] for r in tr)), dict(Counter(r["y"] for r in te))))
    A("")

    # ================= S2 =================
    A("## S2. Guideline-derived specificity features")
    A("")
    A("Each feature corresponds to a phrase the annotation guideline uses to "
      "define quality: *timelines, figures, percentages, identification of "
      "involved parties, clear descriptions of the intended outcome*. Computed "
      "on the gold `evidence_string`.")
    A("")
    bin_tr = np.array([1 if r["y"] == "Clear" else 0 for r in tr])
    keep = np.array([r["y"] in ("Clear", "Not Clear") for r in tr])
    Xtr = feat_matrix(tr, "ev")

    A("| feature | guideline phrase | P(Clear \\| present) | P(Clear \\| absent) | lift | Cramer's V | prevalence |")
    A("|---|---|---:|---:|---:|---:|---:|")
    rows_v = []
    for j, (name, _, phrase) in enumerate(FEATURES):
        f = Xtr[keep, j]
        y = bin_tr[keep]
        p1 = y[f == 1].mean() if (f == 1).any() else float("nan")
        p0 = y[f == 0].mean() if (f == 0).any() else float("nan")
        v = cramers_v(f, y)
        rows_v.append((name, v))
        A("| `%s` | %s | %.3f | %.3f | %+.3f | %.3f | %.1f%% |"
          % (name, phrase, p1, p0, p1 - p0, v, 100.0 * f.mean()))
    A("")
    base_rate = bin_tr[keep].mean()
    A("Base rate P(Clear) among Clear/Not Clear = **%.3f**." % base_rate)
    A("")
    strongest = sorted(rows_v, key=lambda x: -x[1])[:3]
    A("Strongest associations: %s."
      % ", ".join("`%s` V=%.3f" % s for s in strongest))
    A("")

    # features-only classifier, Clear vs Not Clear
    tr2 = [r for r in tr if r["y"] in ("Clear", "Not Clear")]
    te2 = [r for r in te if r["y"] in ("Clear", "Not Clear")]
    Xa = feat_matrix(tr2, "ev")
    Xb = feat_matrix(te2, "ev")
    sc = StandardScaler().fit(Xa)
    ya = np.array([1 if r["y"] == "Clear" else 0 for r in tr2])
    yb = np.array([1 if r["y"] == "Clear" else 0 for r in te2])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=SEED).fit(sc.transform(Xa), ya)
    pb = lr.predict(sc.transform(Xb))
    auc = roc_auc_score(yb, lr.predict_proba(sc.transform(Xb))[:, 1])
    mf1 = f1_score(yb, pb, average="macro", zero_division=0)
    A("**Features-only classifier** (logistic regression on the %d features "
      "above, Clear vs Not Clear, fit on train+val, evaluated on test): "
      "macro-F1 **%.4f**, ROC-AUC **%.4f**." % (Xa.shape[1], mf1, auc))
    A("")
    A("For reference on the same binary problem: always-predict-Clear gives "
      "macro-F1 %.4f."
      % f1_score(yb, np.ones_like(yb), average="macro", zero_division=0))
    A("")

    # ================= S1 =================
    A("## S1. Promise-evidence relatedness (lexical)")
    A("")
    A("The guideline defines Misleading as evidence that *has no clear "
      "connection with the claim*. Relatedness between `promise_string` and "
      "`evidence_string` is therefore a zero-shot signal. Evaluated primarily on "
      "Clear vs Not Clear, where n is large enough to trust.")
    A("")
    both = [r for r in tr if r["pr"]]
    cos = char_tfidf_cos([r["pr"] for r in both], [r["ev"] for r in both])
    jac = np.array([jaccard_char(r["pr"], r["ev"]) for r in both])
    lenr = np.array([len(r["ev"]) / max(1, len(r["pr"])) for r in both])
    ys = [r["y"] for r in both]

    A("| measure | Clear | Not Clear | Misleading | AUC (Clear vs Not Clear) |")
    A("|---|---:|---:|---:|---:|")
    for nm, arr in [("char TF-IDF cosine", cos), ("char bigram Jaccard", jac),
                    ("len(evidence)/len(promise)", lenr)]:
        m = {c: arr[[i for i, y in enumerate(ys) if y == c]] for c in CLASSES3}
        idx = [i for i, y in enumerate(ys) if y in ("Clear", "Not Clear")]
        yy = [1 if ys[i] == "Clear" else 0 for i in idx]
        a = roc_auc_score(yy, arr[idx]) if len(set(yy)) > 1 else float("nan")
        A("| %s | %.4f | %.4f | %s | %.4f |"
          % (nm, m["Clear"].mean(), m["Not Clear"].mean(),
             ("%.4f" % m["Misleading"].mean()) if len(m["Misleading"]) else "n/a", a))
    A("")
    mis_idx = [i for i, y in enumerate(ys) if y == "Misleading"]
    if mis_idx:
        A("Percentile rank of the Misleading instance(s) in the cosine "
          "distribution: %s (0 = least related). n=%d, so this is an "
          "observation, not evidence."
          % (", ".join("%.1f" % (100.0 * (cos < cos[i]).mean()) for i in mis_idx),
             len(mis_idx)))
    A("")

    dest = os.path.join(OUT, "s1s2_diagnostics.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest)
    print()
    print("\n".join(L))


if __name__ == "__main__":
    main()
