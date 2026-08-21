# NTCIR-19 / AI CUP VeriPromise ESG Track — research workspace

Work for the paper submission, kept separate from the competition system code in
`../scripts`.

## Working thesis

VeriPromiseESG4K (Taiwan, 49 companies, 4,000 instances) is a ~5x scale-up of the
Chinese subset of **ML-Promise** (Seki et al., EMNLP 2025), which covered 9
Taiwanese companies. Several conclusions published from the smaller corpus do not
replicate at the larger scale. The paper reports which ones survive, which do not,
and what that implies for benchmark design in this task family.

Task lineage: ML-ESG-1/2/3 (FinNLP 2023-2024) -> ML-Promise (EMNLP 2025) ->
SemEval-2025 Task 6 PromiseEval -> AI CUP VeriPromiseESG (2026) -> NTCIR-19 RegCom.

## Layout

```
research/
  data/
    raw/twse_listed_companies.json   TWSE OpenAPI t187ap03_L (industry, capital)
    raw/twse_stock_day_all.json      TWSE OpenAPI STOCK_DAY_ALL (closing prices)
    company_metadata.csv             49 tickers x industry, market cap, tiers
    mlpromise_reference.json         ML-Promise published figures, hand-transcribed
                                     with table citations + the 6 testable claims
  scripts/
    01_build_company_metadata.py     joins the dataset tickers to TWSE metadata
    02_distribution_comparison.py    the step 1-2 comparison and statistical tests
  output/
    step12_distribution_comparison.md   main report
    industry_comparison.csv
    marketcap_comparison.csv
```

## Reproduce

```bash
python research/scripts/01_build_company_metadata.py
python research/scripts/02_distribution_comparison.py
```

Requires `numpy` and `scipy` (already in the project env) plus `pypdf` for reading
the reference PDFs. The two `raw/*.json` files are committed so the analysis is
reproducible without re-hitting the TWSE API; regenerate with:

```bash
curl -s "https://openapi.twse.com.tw/v1/opendata/t187ap03_L" -o research/data/raw/twse_listed_companies.json
curl -s "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL" -o research/data/raw/twse_stock_day_all.json
```

## Status

- [x] Step 1 — company metadata (industry, market-cap tiers) from TWSE OpenAPI
- [x] Step 2 — distribution comparison vs ML-Promise, industry- and market-cap-controlled
- [ ] Step 3 — document-level evidence retrieval (needs the 49 report PDFs; blocked on disk space)
- [ ] Step 4 — structural-consistency / benchmark-defect section
- [ ] Step 5 — draft

## Headline results so far

See `output/step12_distribution_comparison.md`. In short:

- Of five testable published claims, **four are refuted and one replicates**
  (Semiconductor >5-year commitments: 38.85% published vs 37.2% measured).
- The largest divergence is `within_2_years`: **37.5% -> 2.1%**. Controlling for
  industry does not close it (Semiconductor: 59.87% -> 7.1%).
- Crucially, the timing comparison is conditional on a promise existing, so it is
  **not** explained by the corpus sampling more promise-bearing paragraphs — which
  is the obvious confound for the promise-rate and evidence-rate gaps.
- Three concrete benchmark defects surfaced: the train/val `""` vs `"N/A"` encoding
  split, a company-name casing collision that inflates 49 companies to 50, and
  inconsistent `esg_type` multi-label ordering.
