"""Compare VeriPromiseESG4K label distributions against the published ML-Promise
Chinese (Taiwan) subset, overall and controlling for industry / market cap.

Outputs a markdown report plus machine-readable CSVs under research/output/.

Label alignment with ML-Promise (Seki et al., 2025)
---------------------------------------------------
  promise_status        <-> Promise Identification (PI)
  evidence_status       <-> Actionable Evidence (AE)
  evidence_quality      <-> Clarity of the Promise-Evidence Pair (CPEP)
  verification_timeline <-> Timing for Verification (TV)

ML-Promise Table 1 reports AE over *all* instances (so AE=No absorbs the
promise=No cases). VeriPromiseESG4K instead marks those N/A, so for a
like-for-like comparison we collapse {N/A, ""} into "No" when reproducing the
Table 1 view, and separately report the conditional view (AE given PI=Yes).

ML-Promise Tables 2 and 3 renormalise TV over the three real horizons and drop
the "Other" class, so the industry / market-cap comparisons do the same.
"""
import csv
import io
import json
import os
import sys
from collections import Counter, OrderedDict

import numpy as np
from scipy.stats import chi2_contingency, chisquare

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESEARCH = os.path.join(ROOT, "research")
OUT = os.path.join(RESEARCH, "output")
os.makedirs(OUT, exist_ok=True)

NA_FORMS = {"N/A", ""}
TIMING_REAL = ["within_2_years", "between_2_and_5_years", "more_than_5_years"]
TIMING_ALL = TIMING_REAL + ["already"]
QUALITY_REAL = ["Clear", "Not Clear", "Misleading"]

# VeriPromiseESG4K -> ML-Promise timing label names
TIMING_ALIAS = {"within_2_years": "within_2_years",
                "between_2_and_5_years": "between_2_and_5_years",
                "more_than_5_years": "more_than_5_years",
                "already": "other"}


def load_labeled():
    """Return the 2,000 labelled rows (train + val); test has no gold labels."""
    rows = []
    for fn, split in [("vpesg_4k_train_1000.json", "train"),
                      ("vpesg4k_val_1000.json", "val")]:
        d = json.load(open(os.path.join(ROOT, "data", fn), encoding="utf-8"))
        if isinstance(d, dict):
            d = d.get("data", list(d.values())[0])
        for r in d:
            r["_split"] = split
        rows += d
    return rows


def load_meta():
    path = os.path.join(RESEARCH, "data", "company_metadata.csv")
    return {r["ticker"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def pct(counter, keys):
    tot = sum(counter.get(k, 0) for k in keys)
    if tot == 0:
        return OrderedDict((k, 0.0) for k in keys), 0
    return OrderedDict((k, 100.0 * counter.get(k, 0) / tot) for k in keys), tot


def vp_distributions(rows):
    """Compute the four VeriPromiseESG4K distributions in ML-Promise form."""
    d = {}

    pi = Counter(r["promise_status"] for r in rows)
    d["promise_identification"] = pct(pi, ["Yes", "No"])

    # Table 1 convention: AE over all instances, N/A collapsed into No.
    ae_all = Counter("Yes" if r["evidence_status"] == "Yes" else "No" for r in rows)
    d["actionable_evidence_table1"] = pct(ae_all, ["Yes", "No"])

    # Conditional convention: AE given a promise exists.
    pyes = [r for r in rows if r["promise_status"] == "Yes"]
    ae_cond = Counter(r["evidence_status"] for r in pyes)
    d["actionable_evidence_conditional"] = pct(ae_cond, ["Yes", "No"])

    q = Counter(r["evidence_quality"] for r in rows
                if r["evidence_quality"] not in NA_FORMS)
    d["clarity"] = pct(q, QUALITY_REAL)

    tv_all = Counter(TIMING_ALIAS.get(r["verification_timeline"], r["verification_timeline"])
                     for r in rows if r["verification_timeline"] not in NA_FORMS)
    d["timing_with_other"] = pct(tv_all, ["within_2_years", "between_2_and_5_years",
                                          "more_than_5_years", "other"])
    tv_real = Counter(r["verification_timeline"] for r in rows
                      if r["verification_timeline"] in TIMING_REAL)
    d["timing_real_only"] = pct(tv_real, TIMING_REAL)
    return d


def two_sample_chi2(counts_a, counts_b, keys):
    """Chi-square test of homogeneity between two observed count vectors."""
    a = [counts_a.get(k, 0) for k in keys]
    b = [counts_b.get(k, 0) for k in keys]
    table = np.array([a, b], dtype=float)
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        return None
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof),
            "n": int(n), "cramers_v": v}


def gof_vs_proportions(observed_counts, ref_pct, keys):
    """Goodness-of-fit of observed counts against published reference percentages."""
    obs = np.array([observed_counts.get(k, 0) for k in keys], dtype=float)
    exp_p = np.array([ref_pct.get(k, 0.0) for k in keys], dtype=float)
    if exp_p.sum() == 0 or obs.sum() == 0:
        return None
    exp_p = exp_p / exp_p.sum()
    # Zero expected cells make the statistic undefined; floor them at 0.5 counts.
    exp = exp_p * obs.sum()
    floor = 0.5
    if (exp < floor).any():
        exp = np.maximum(exp, floor)
        exp = exp / exp.sum() * obs.sum()
    chi2, p = chisquare(obs, exp)
    return {"chi2": float(chi2), "p": float(p), "dof": len(keys) - 1,
            "n": int(obs.sum())}


def fmt_p(p):
    if p is None:
        return "n/a"
    return "< 1e-300" if p == 0 else ("%.3g" % p)


def counter_for(rows, field, keys, alias=None):
    c = Counter()
    for r in rows:
        v = r[field]
        if alias:
            v = alias.get(v, v)
        if v in keys:
            c[v] += 1
    return c


def main():
    rows = load_labeled()
    meta = load_meta()
    ref = json.load(open(os.path.join(RESEARCH, "data", "mlpromise_reference.json"),
                         encoding="utf-8"))

    for r in rows:
        m = meta.get(str(r["ticker"]).strip(), {})
        r["_industry"] = m.get("industry_en", "UNKNOWN")
        r["_group"] = m.get("mlpromise_group", "Other")
        r["_tier"] = m.get("mktcap_tier_global", "")
        r["_tier_ind"] = m.get("mktcap_tier_within_industry", "")

    vp = vp_distributions(rows)
    t1 = ref["table1_chinese"]
    lines = []
    A = lines.append

    A("# Step 1-2: VeriPromiseESG4K vs ML-Promise (Chinese / Taiwan) distribution comparison")
    A("")
    A("Generated by `research/scripts/02_distribution_comparison.py`.")
    A("")
    A("- **VeriPromiseESG4K**: %d labelled instances (train 1,000 + val 1,000), "
      "49 unique tickers, 49 ESG reports." % len(rows))
    A("- **ML-Promise Chinese**: 9 Taiwanese companies across 3 industries; "
      "Table 1 counts as published.")
    A("- Test set (2,000 instances) is excluded throughout: no gold labels were released.")
    A("")
    A("> ML-Promise Table 1 reports 199 instances in the Chinese Promise-Identification "
      "row while Section 3.4 states 610 Chinese instances. The published counts are used "
      "as-is for the homogeneity tests; this discrepancy is a caveat on those p-values, "
      "not on the percentages.")
    A("")

    # ---------------- Table 1 comparison ----------------
    A("## 1. Overall distributions (ML-Promise Table 1 view)")
    A("")
    A("| Task | Label | ML-Promise ZH | VeriPromiseESG4K | Delta (pp) |")
    A("|---|---|---:|---:|---:|")

    rowsets = [
        ("Promise Identification", "promise_identification", ["Yes", "No"],
         t1["promise_identification"]),
        ("Actionable Evidence", "actionable_evidence_table1", ["Yes", "No"],
         t1["actionable_evidence"]),
        ("Clarity (CPEP)", "clarity", QUALITY_REAL, t1["clarity"]),
        ("Timing (TV)", "timing_with_other",
         ["within_2_years", "between_2_and_5_years", "more_than_5_years", "other"],
         t1["timing"]),
    ]
    tests = []
    for label, key, keys, refblock in rowsets:
        vp_pct, vp_tot = vp[key]
        refc = {k: refblock[k] for k in keys if k in refblock}
        ref_tot = sum(refc.values())
        vp_counts = Counter()
        for k in keys:
            vp_counts[k] = round(vp_pct[k] / 100.0 * vp_tot)
        for k in keys:
            rp = 100.0 * refc.get(k, 0) / ref_tot if ref_tot else 0.0
            A("| %s | `%s` | %.1f%% (n=%d) | %.1f%% (n=%d) | %+.1f |"
              % (label, k, rp, refc.get(k, 0), vp_pct[k], vp_counts[k], vp_pct[k] - rp))
        t = two_sample_chi2(refc, vp_counts, keys)
        if t:
            tests.append((label, t))
    A("")
    A("### Homogeneity tests (ML-Promise ZH vs VeriPromiseESG4K)")
    A("")
    A("| Task | chi2 | dof | p | Cramer's V | n |")
    A("|---|---:|---:|---:|---:|---:|")
    for label, t in tests:
        A("| %s | %.1f | %d | %s | %.3f | %d |"
          % (label, t["chi2"], t["dof"], fmt_p(t["p"]), t["cramers_v"], t["n"]))
    A("")

    A("### Conditional views (VeriPromiseESG4K)")
    A("")
    ae_c, ae_n = vp["actionable_evidence_conditional"]
    A("- Actionable Evidence **given promise_status=Yes**: "
      + ", ".join("`%s` %.1f%%" % (k, v) for k, v in ae_c.items())
      + " (n=%d)." % ae_n)
    A("  ML-Promise equivalent is 40/80 = **50.0%** Yes.")
    tr, trn = vp["timing_real_only"]
    A("- Timing **excluding the already/Other class** (the convention used in "
      "ML-Promise Tables 2-3): "
      + ", ".join("`%s` %.1f%%" % (k, v) for k, v in tr.items()) + " (n=%d)." % trn)
    A("")

    # ---------------- Industry-controlled ----------------
    A("## 2. Industry-controlled comparison (ML-Promise Table 3 view)")
    A("")
    A("ML-Promise's Chinese subset covers Semiconductor, Energy (Oil/Gas/Electricity) "
      "and Biomedical only. Restricting VeriPromiseESG4K to the same TWSE industries "
      "tests whether the overall gap is driven by industry composition.")
    A("")
    ind_csv = [["industry", "task", "label", "mlpromise_pct", "vpesg_pct",
                "vpesg_n", "n_companies"]]
    for grp in ["Semiconductor", "Energy", "Biomedical"]:
        sub = [r for r in rows if r["_group"] == grp]
        ncomp = len(set(r["ticker"] for r in sub))
        A("### %s  (VeriPromiseESG4K: %d instances, %d companies)"
          % (grp, len(sub), ncomp))
        A("")
        if not sub:
            A("_No instances._")
            A("")
            continue
        rb = ref["table3_chinese_by_industry"][grp]
        A("| Task | Label | ML-Promise ZH | VeriPromiseESG4K | Delta (pp) |")
        A("|---|---|---:|---:|---:|")
        specs = [
            ("Promise Id.", "promise_status", ["Yes", "No"], None,
             rb["promise_identification"]),
            ("Act. Evidence", "_ae", ["Yes", "No"], None, rb["actionable_evidence"]),
            ("Clarity", "evidence_quality", QUALITY_REAL, None, rb["clarity"]),
            ("Timing", "verification_timeline", TIMING_REAL, None, rb["timing"]),
        ]
        for tname, field, keys, alias, refpct in specs:
            if field == "_ae":
                c = Counter("Yes" if r["evidence_status"] == "Yes" else "No" for r in sub)
            else:
                c = counter_for(sub, field, keys, alias)
            p_, n_ = pct(c, keys)
            for k in keys:
                A("| %s | `%s` | %.2f%% | %.1f%% (n=%d) | %+.1f |"
                  % (tname, k, refpct.get(k, 0.0), p_[k], c.get(k, 0),
                     p_[k] - refpct.get(k, 0.0)))
                ind_csv.append([grp, tname, k, refpct.get(k, 0.0),
                                round(p_[k], 2), c.get(k, 0), ncomp])
            g = gof_vs_proportions(c, refpct, keys)
            if g:
                A("| %s | _GOF vs ML-Promise_ | | chi2=%.1f, dof=%d, p=%s | |"
                  % (tname, g["chi2"], g["dof"], fmt_p(g["p"])))
        A("")

    # ---------------- Market cap ----------------
    A("## 3. Market-capitalisation comparison (ML-Promise Table 2 view)")
    A("")
    A("Tiers are global tertiles over the 49 companies by market cap "
      "(TWSE paid-in capital / par value x closing price, snapshot ROC "
      + meta[list(meta)[0]]["price_snapshot_roc_date"] + ").")
    A("")
    cap_csv = [["tier", "task", "label", "mlpromise_pct", "vpesg_pct", "vpesg_n",
                "n_companies"]]
    for tier in ["High", "Medium", "Low"]:
        sub = [r for r in rows if r["_tier"] == tier]
        ncomp = len(set(r["ticker"] for r in sub))
        rb = ref["table2_chinese_by_marketcap"][tier]
        A("### %s market cap  (VeriPromiseESG4K: %d instances, %d companies)"
          % (tier, len(sub), ncomp))
        A("")
        A("| Task | Label | ML-Promise ZH | VeriPromiseESG4K | Delta (pp) |")
        A("|---|---|---:|---:|---:|")
        specs = [
            ("Promise Id.", "promise_status", ["Yes", "No"], rb["promise_identification"]),
            ("Act. Evidence", "_ae", ["Yes", "No"], rb["actionable_evidence"]),
            ("Clarity", "evidence_quality", QUALITY_REAL, rb["clarity"]),
            ("Timing", "verification_timeline", TIMING_REAL, rb["timing"]),
        ]
        for tname, field, keys, refpct in specs:
            if field == "_ae":
                c = Counter("Yes" if r["evidence_status"] == "Yes" else "No" for r in sub)
            else:
                c = counter_for(sub, field, keys)
            p_, n_ = pct(c, keys)
            for k in keys:
                A("| %s | `%s` | %.2f%% | %.1f%% (n=%d) | %+.1f |"
                  % (tname, k, refpct.get(k, 0.0), p_[k], c.get(k, 0),
                     p_[k] - refpct.get(k, 0.0)))
                cap_csv.append([tier, tname, k, refpct.get(k, 0.0),
                                round(p_[k], 2), c.get(k, 0), ncomp])
        A("")

    # ---------------- Monotonicity of published claims ----------------
    A("## 4. Direct tests of the published ML-Promise claims")
    A("")
    A("| ID | Claim (abbreviated) | Prediction | VeriPromiseESG4K | Verdict |")
    A("|---|---|---|---|---|")

    tiers = ["High", "Medium", "Low"]
    pi_yes = []
    w2 = []
    clear = []
    for tier in tiers:
        sub = [r for r in rows if r["_tier"] == tier]
        c = Counter(r["promise_status"] for r in sub)
        pi_yes.append(100.0 * c["Yes"] / max(1, c["Yes"] + c["No"]))
        ct = counter_for(sub, "verification_timeline", TIMING_REAL)
        tot = sum(ct.values())
        w2.append(100.0 * ct["within_2_years"] / max(1, tot))
        cq = counter_for(sub, "evidence_quality", QUALITY_REAL)
        clear.append(100.0 * cq["Clear"] / max(1, sum(cq.values())))

    vp_t, vp_tn = vp["timing_real_only"]
    verdict_c1 = "REFUTED" if vp_t["within_2_years"] < 10 else "SUPPORTED"
    A("| C1 | Taiwan firms favour short-term (<2y) promises | `within_2_years` "
      "should be frequent | **%.1f%%** of promise-bearing instances (excl. already), "
      "**%.1f%%** including it | **%s** |"
      % (vp_t["within_2_years"], vp["timing_with_other"][0]["within_2_years"], verdict_c1))
    mono_dec = w2[0] < w2[1] < w2[2]
    A("| C2 | `within_2_years` decreases with market cap | High < Med < Low | "
      "High %.1f%%, Med %.1f%%, Low %.1f%% | **%s** |"
      % (w2[0], w2[1], w2[2], "SUPPORTED" if mono_dec else "REFUTED"))
    mono_inc = pi_yes[0] > pi_yes[1] > pi_yes[2]
    A("| C3 | promise=Yes rate increases with market cap | High > Med > Low | "
      "High %.1f%%, Med %.1f%%, Low %.1f%% | **%s** |"
      % (pi_yes[0], pi_yes[1], pi_yes[2], "SUPPORTED" if mono_inc else "REFUTED"))
    lowest_clear = clear.index(max(clear)) == 2
    A("| C4 | Clarity highest for low-cap firms | Low tier highest `Clear` | "
      "High %.1f%%, Med %.1f%%, Low %.1f%% | **%s** |"
      % (clear[0], clear[1], clear[2], "SUPPORTED" if lowest_clear else "REFUTED"))
    semi = [r for r in rows if r["_group"] == "Semiconductor"]
    sc = counter_for(semi, "verification_timeline", TIMING_REAL)
    stot = max(1, sum(sc.values()))
    semi_gt5 = 100.0 * sc["more_than_5_years"] / stot
    A("| C5 | Semiconductor: 38.85%% are >5y commitments | ~39%% | **%.1f%%** "
      "(n=%d, %d companies) | **%s** |"
      % (semi_gt5, stot, len(set(r["ticker"] for r in semi)),
         "SUPPORTED" if abs(semi_gt5 - 38.85) < 10 else "REFUTED"))
    A("")

    # ---------------- Robustness ----------------
    A("## 5. Robustness of the C1 refutation")
    A("")
    A("If `within_2_years` were rare only because a handful of companies dominate the "
      "corpus, the refutation would be an artefact of company weighting. Per-company "
      "rates (among promise-bearing instances, excluding `already`):")
    A("")
    per_comp = {}
    for r in rows:
        if r["verification_timeline"] in TIMING_REAL:
            t = str(r["ticker"]).strip()
            per_comp.setdefault(t, Counter())[r["verification_timeline"]] += 1
    rates = []
    for t, c in per_comp.items():
        tot = sum(c.values())
        rates.append((t, 100.0 * c["within_2_years"] / tot, tot, c["within_2_years"]))
    rates.sort(key=lambda x: -x[1])
    n_any = sum(1 for _, _, _, w in rates if w > 0)
    arr = np.array([x[1] for x in rates])
    A("- Companies with >=1 promise-bearing instance in the three real horizons: **%d**"
      % len(rates))
    A("- Companies with **at least one** `within_2_years` instance: **%d** (%.0f%%)"
      % (n_any, 100.0 * n_any / len(rates)))
    A("- Per-company `within_2_years` rate: median **%.1f%%**, mean **%.1f%%**, "
      "max **%.1f%%** (ticker `%s`)" % (np.median(arr), arr.mean(), arr[0], rates[0][0]))
    A("- Companies exceeding the ML-Promise Semiconductor figure of 59.87%%: **%d**"
      % sum(1 for x in rates if x[1] > 59.87))
    A("")
    A("| ticker | within_2_years | n (3 horizons) | rate |")
    A("|---|---:|---:|---:|")
    for t, rt, tot, w in rates[:8]:
        A("| `%s` | %d | %d | %.1f%% |" % (t, w, tot, rt))
    A("| ... | | | |")
    A("")
    A("The label is rare *within nearly every company*, so the gap is a corpus-level "
      "property rather than a company-weighting artefact.")
    A("")

    # ---------------- Threats to validity ----------------
    A("## 6. Threats to validity")
    A("")
    A("**T1. Differential paragraph sampling (the main confound).** VeriPromiseESG4K "
      "shows an 81.3% promise rate against 40.2% in ML-Promise, which is consistent "
      "with the competition corpus having preferentially sampled promise-bearing "
      "paragraphs. This alone could explain the Promise-Identification and "
      "Actionable-Evidence gaps and they should not be read as evidence about "
      "corporate behaviour.")
    A("")
    A("  _However, it does not explain the timing result._ The `within_2_years` "
      "comparison is computed **conditional on a promise existing** in both corpora, "
      "so over-sampling promise-bearing paragraphs leaves it unaffected. The C1 "
      "refutation is therefore substantially more robust than the C3/C4 ones.")
    A("")
    A("**T2. Small expected cells in the goodness-of-fit tests.** The Semiconductor "
      "timing GOF is dominated by the `between_2_and_5_years` cell, whose ML-Promise "
      "proportion of 1.27% yields an expected count of ~1.4 against 63 observed. "
      "That is below the conventional expected-count-of-5 threshold, so the chi2 "
      "magnitude is inflated even though the direction and effect size are "
      "unambiguous. Report the effect size, not the statistic.")
    A("")
    A("**T3. Single-company industry cells.** The Energy and Biomedical groups contain "
      "one company each (27 instances). Only the Semiconductor comparison (7 companies, "
      "305 instances) supports an industry-level claim.")
    A("")
    A("**T4. Label-vocabulary drift.** ML-Promise uses `Other` where VeriPromiseESG4K "
      "uses `already`; the guideline text for both is 'already verified or no specific "
      "timing'. Their shares are close (37.5% vs 44.1%), so the timing shift is a "
      "movement *between the three real horizons*, not an artefact of this rename.")
    A("")
    A("**T5. Annotator and guideline revision.** The two corpora were annotated by "
      "different teams. ML-Promise reports Cohen's kappa 0.62-0.80 for CPEP and "
      "0.60-0.89 for TV; no agreement figures have been published for "
      "VeriPromiseESG4K. Part of the gap may be annotator drift rather than corpus "
      "content, which is itself a finding worth reporting.")
    A("")
    A("**T6. Market-cap snapshot.** Tiers use a current TWSE price snapshot, not the "
      "value at 2024 report publication. Tier boundaries could shift for companies "
      "near a tertile edge.")
    A("")

    # ---------------- Data quality notes ----------------
    A("## 7. Data-quality observations surfaced by this comparison")
    A("")
    tr_na = Counter()
    for r in rows:
        if r["promise_status"] == "No":
            tr_na[(r["_split"], repr(r["evidence_status"]))] += 1
    A("**(a) The two labelled splits encode the not-applicable state differently.**")
    A("")
    A("| split | promise_status=No rows | `evidence_status` surface form |")
    A("|---|---:|---|")
    for (split, form), n in sorted(tr_na.items()):
        A("| %s | %d | %s |" % (split, n, form))
    A("")
    A("The official `sample_submission_format.csv` uses `N/A`. A model trained "
      "directly on the train split without normalisation emits the empty string and "
      "scores zero on every not-applicable row. The same split applies to "
      "`verification_timeline` and `evidence_quality`.")
    A("")

    name_coll = [m for m in meta.values() if len(m["name_variants"].split("|")) > 1]
    A("**(b) The nominal 50 companies are 49.** ")
    for m in name_coll:
        A("Ticker `%s` appears under %d name variants (`%s`), inflating a naive "
          "company count to 50." % (m["ticker"], len(m["name_variants"].split("|")),
                                    m["name_variants"]))
    A("")

    esg = Counter(r.get("esg_type", "") for r in rows)
    multi = {k: v for k, v in esg.items() if ";" in k}
    A("**(c) `esg_type` multi-labels use inconsistent orderings.** "
      "%d distinct multi-label strings covering %d instances: %s. Empty: %d."
      % (len(multi), sum(multi.values()),
         ", ".join("`%s`=%d" % (k, v) for k, v in sorted(multi.items())),
         esg.get("", 0)))
    A("")

    mis = sum(1 for r in rows if r["evidence_quality"] == "Misleading")
    A("**(d) `Misleading` support is n=%d** in VeriPromiseESG4K and n=0 in the "
      "ML-Promise Chinese subset (10 across all five ML-Promise languages combined). "
      "It carries 1/4 of the `evidence_quality` macro-F1 weight, which itself is 35%% "
      "of the official score." % mis)
    A("")

    report = os.path.join(OUT, "step12_distribution_comparison.md")
    open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    with open(os.path.join(OUT, "industry_comparison.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(ind_csv)
    with open(os.path.join(OUT, "marketcap_comparison.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(cap_csv)

    print("wrote", report)
    print("wrote", os.path.join(OUT, "industry_comparison.csv"))
    print("wrote", os.path.join(OUT, "marketcap_comparison.csv"))
    print()
    print("\n".join(lines[:0]))


if __name__ == "__main__":
    main()
