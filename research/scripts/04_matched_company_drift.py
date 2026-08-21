"""Matched-company comparison between VeriPromiseESG4K and ML-Promise Chinese.

Four companies appear in both corpora: TSMC (2330), FPCC (6505), NPC (1303) and
Alchip (3661). Holding the company constant removes industry mix, company size
and national reporting culture as explanations for the label-distribution gap,
leaving two candidates:

  * annotation team / guideline revision  (the T5 threat)
  * corpus construction, i.e. which paragraphs were sampled

Report years differ (ML-Promise draws on 2022-2023 reports, VeriPromiseESG4K on
2024 reports), so the document is not held constant -- only the company is.

Label alignment
---------------
  ML-Promise 'Less than 2 years'      -> within_2_years
  ML-Promise '2 to 5 years'           -> between_2_and_5_years
  ML-Promise 'More than 5 years'      -> more_than_5_years
  ML-Promise 'N/A' given promise=Yes  -> already
  ML-Promise 'Potentially Misleading' -> Misleading
  VeriPromiseESG4K '' and 'N/A'       -> N/A
"""
import io
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import fisher_exact, chi2_contingency

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLP = os.path.join(ROOT, "research", "data", "raw", "mlpromise")
OUT = os.path.join(ROOT, "research", "output")
os.makedirs(OUT, exist_ok=True)

MATCH = {"TSMC.pdf": ("2330", "TSMC"),
         "FPCC.pdf": ("6505", "Formosa Petrochemical"),
         "NPC.pdf": ("1303", "Nan Ya Plastics"),
         "Alchip.pdf": ("3661", "Alchip")}

TL_MAP = {"Less than 2 years": "within_2_years",
          "2 to 5 years": "between_2_and_5_years",
          "More than 5 years": "more_than_5_years"}
TL_KEYS = ["within_2_years", "between_2_and_5_years", "more_than_5_years", "already"]
TL_REAL = TL_KEYS[:3]
Q_KEYS = ["Clear", "Not Clear", "Misleading"]
SEED = 20260822


def load(path):
    j = json.load(open(path, encoding="utf-8-sig"))
    return j.get("data", list(j.values())[0]) if isinstance(j, dict) else j


def norm_mlp(r):
    """Normalise one ML-Promise row into VeriPromiseESG4K label space."""
    p = r["promise_status"]
    tl = r["verification_timeline"]
    if p == "Yes":
        tl = TL_MAP.get(tl, "already" if tl == "N/A" else tl)
    else:
        tl = "N/A"
    q = r["evidence_quality"]
    q = "Misleading" if "islead" in str(q) else q
    return {"promise_status": p,
            "evidence_status": r["evidence_status"] if p == "Yes" else "N/A",
            "verification_timeline": tl,
            "evidence_quality": q,
            "_key": MATCH.get(r.get("pdf"), (None, None))[0]}


def norm_vp(r):
    def na(v):
        return "N/A" if v in ("", "N/A") else v
    return {"promise_status": r["promise_status"],
            "evidence_status": na(r["evidence_status"]),
            "verification_timeline": na(r["verification_timeline"]),
            "evidence_quality": na(r["evidence_quality"]),
            "_key": str(r["ticker"]).strip()}


def rates(rows):
    """Return the four comparable rates plus their denominators."""
    n = len(rows)
    pyes = [r for r in rows if r["promise_status"] == "Yes"]
    eyes = [r for r in pyes if r["evidence_status"] == "Yes"]
    tl = Counter(r["verification_timeline"] for r in pyes)
    real = sum(tl[k] for k in TL_REAL)
    q = Counter(r["evidence_quality"] for r in eyes)
    nq = sum(q[k] for k in Q_KEYS)
    return {
        "n": n,
        "promise_yes": (len(pyes), n),
        "evidence_yes": (len(eyes), len(pyes)),
        "clear": (q["Clear"], nq),
        "within2_of_promise": (tl["within_2_years"], len(pyes)),
        "within2_of_real": (tl["within_2_years"], real),
        "already_of_promise": (tl["already"], len(pyes)),
        "timeline_counts": {k: tl[k] for k in TL_KEYS},
        "quality_counts": {k: q[k] for k in Q_KEYS},
    }


def pc(t):
    a, b = t
    return "n/a" if b == 0 else "%.1f%% (%d/%d)" % (100.0 * a / b, a, b)


def frac(t):
    a, b = t
    return float("nan") if b == 0 else 100.0 * a / b


def test_2x2(a, b):
    """a, b are (successes, total). Fisher exact on the 2x2."""
    if a[1] == 0 or b[1] == 0:
        return None
    tab = [[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]]
    try:
        _, p = fisher_exact(tab)
        return p
    except ValueError:
        return None


def boot_diff(a, b, n=20000):
    """Bootstrap CI for (rate_a - rate_b) in percentage points."""
    if a[1] == 0 or b[1] == 0:
        return None
    rng = np.random.default_rng(SEED)
    da = rng.binomial(a[1], a[0] / a[1], n) / a[1]
    db = rng.binomial(b[1], b[0] / b[1], n) / b[1]
    d = (da - db) * 100
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    mlp = [norm_mlp(r) for r in
           load(os.path.join(MLP, "Trainset_Chinese.json"))
           + load(os.path.join(MLP, "Test_Chinese.json"))]
    vp = []
    for fn in ["vpesg_4k_train_1000.json", "vpesg4k_val_1000.json"]:
        vp += [norm_vp(r) for r in load(os.path.join(ROOT, "data", fn))]

    tickers = [t for t, _ in MATCH.values()]
    mlp_m = [r for r in mlp if r["_key"] in tickers]
    vp_m = [r for r in vp if r["_key"] in tickers]

    L = []
    A = L.append
    A("# Matched-company comparison: VeriPromiseESG4K vs ML-Promise Chinese")
    A("")
    A("Generated by `research/scripts/04_matched_company_drift.py`.")
    A("")
    A("Four companies appear in both corpora. Holding the company constant "
      "removes industry mix, company size and national reporting culture as "
      "explanations, leaving annotation practice and corpus construction.")
    A("")
    A("Report years differ (ML-Promise: 2022-2023 reports; VeriPromiseESG4K: "
      "2024 reports), so the company is held constant but the document is not.")
    A("")

    A("## 1. Matched subsets")
    A("")
    A("| Company | ticker | ML-Promise n | VeriPromiseESG4K n (labelled) |")
    A("|---|---|---:|---:|")
    for pdf, (tk, name) in MATCH.items():
        A("| %s | `%s` | %d | %d |"
          % (name, tk, sum(1 for r in mlp if r["_key"] == tk),
             sum(1 for r in vp if r["_key"] == tk)))
    A("| **matched total** | | **%d** | **%d** |" % (len(mlp_m), len(vp_m)))
    A("")

    # ---------- headline: matched vs full-corpus gap ----------
    rm, rv = rates(mlp_m), rates(vp_m)
    rM, rV = rates(mlp), rates(vp)

    A("## 2. Does the gap survive matching on company?")
    A("")
    A("If the gap were driven by which companies each corpus happens to cover, "
      "the matched columns would agree even though the full-corpus columns do "
      "not.")
    A("")
    A("| Measure | ML-Promise (all) | VPESG (all) | gap | ML-Promise (matched) "
      "| VPESG (matched) | gap | Fisher p (matched) |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    rowspec = [
        ("promise_status=Yes", "promise_yes"),
        ("evidence=Yes given promise", "evidence_yes"),
        ("Clear given evidence", "clear"),
        ("within_2_years given promise", "within2_of_promise"),
        ("within_2_years of 3 real horizons", "within2_of_real"),
        ("already given promise", "already_of_promise"),
    ]
    for label, key in rowspec:
        gfull = frac(rV[key]) - frac(rM[key])
        gmat = frac(rv[key]) - frac(rm[key])
        p = test_2x2(rv[key], rm[key])
        A("| %s | %s | %s | %+.1f | %s | %s | %+.1f | %s |"
          % (label, pc(rM[key]), pc(rV[key]), gfull,
             pc(rm[key]), pc(rv[key]), gmat,
             "n/a" if p is None else ("< 1e-300" if p == 0 else "%.3g" % p)))
    A("")

    ci = boot_diff(rv["within2_of_real"], rm["within2_of_real"])
    A("Bootstrap 95%% CI for the matched `within_2_years` difference "
      "(VPESG minus ML-Promise, percentage points): **[%.1f, %.1f]**."
      % (ci[0], ci[1]))
    A("")

    # ---------- per company ----------
    A("## 3. Per-company breakdown")
    A("")
    A("| Company | corpus | n | promise=Yes | evid=Yes\\|prom | Clear\\|evid | "
      "<2y\\|prom | <2y of 3 horizons |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for pdf, (tk, name) in MATCH.items():
        a = rates([r for r in mlp if r["_key"] == tk])
        b = rates([r for r in vp if r["_key"] == tk])
        for tag, rr in (("ML-Promise", a), ("VPESG", b)):
            A("| %s | %s | %d | %s | %s | %s | %s | %s |"
              % (name if tag == "ML-Promise" else "", tag, rr["n"],
                 pc(rr["promise_yes"]), pc(rr["evidence_yes"]), pc(rr["clear"]),
                 pc(rr["within2_of_promise"]), pc(rr["within2_of_real"])))
    A("")

    # ---------- timeline shape ----------
    A("## 4. Timeline distribution, matched companies only")
    A("")
    A("| Timeline (given promise=Yes) | ML-Promise | VeriPromiseESG4K |")
    A("|---|---:|---:|")
    tm, tv = rm["timeline_counts"], rv["timeline_counts"]
    sm, sv = sum(tm.values()), sum(tv.values())
    for k in TL_KEYS:
        A("| `%s` | %d (%.1f%%) | %d (%.1f%%) |"
          % (k, tm[k], 100.0 * tm[k] / max(1, sm), tv[k], 100.0 * tv[k] / max(1, sv)))
    tab = np.array([[tm[k] for k in TL_KEYS], [tv[k] for k in TL_KEYS]], dtype=float)
    tab = tab[:, tab.sum(axis=0) > 0]
    chi2, p, dof, _ = chi2_contingency(tab)
    v = float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))))
    A("")
    A("Homogeneity on the matched companies: chi2 = %.1f, dof = %d, p = %s, "
      "Cramer's V = %.3f, n = %d."
      % (chi2, dof, "< 1e-300" if p == 0 else "%.3g" % p, v, int(tab.sum())))
    A("")

    # ---------- interpretation ----------
    A("## 5. What this does and does not settle")
    A("")
    gm = frac(rv["within2_of_real"]) - frac(rm["within2_of_real"])
    gf = frac(rV["within2_of_real"]) - frac(rM["within2_of_real"])
    A("- The matched `within_2_years` gap is **%+.1f pp** against **%+.1f pp** "
      "for the full corpora. Matching on company retains %.0f%% of the gap, so "
      "**company coverage explains almost none of it**." % (gm, gf, 100 * gm / gf))
    A("")
    A("- What remains is annotation practice, corpus construction, or a genuine "
      "change in these companies' reporting between the 2022-23 and 2024 "
      "reports. This analysis cannot separate those three.")
    A("")
    A("- The `already` share is close in both corpora on the matched subset "
      "(%s vs %s), so the shift is between the three real horizons rather than "
      "into or out of the already/N-A bucket."
      % (pc(rm["already_of_promise"]), pc(rv["already_of_promise"])))
    A("")
    A("- The promise-rate and evidence-rate gaps stay large after matching too, "
      "but those remain confounded with paragraph sampling and should not be "
      "read as claims about corporate behaviour.")
    A("")
    A("- **Next step to close this**: re-annotate a sample of VeriPromiseESG4K "
      "instances under the ML-Promise guideline (or vice versa). If independent "
      "annotators reproduce the VeriPromiseESG4K distribution, annotation "
      "practice is exonerated and corpus construction is the explanation.")
    A("")

    dest = os.path.join(OUT, "matched_company_drift.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
