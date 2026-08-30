# Codex review

## Bottom line

The project has one clean paper in it, but not the paper the 30 August draft is trying to write.

The defensible contribution is: **VeriPromiseESG4K's unscored promise and evidence spans define a useful extraction task, and the right reduction for this Traditional Chinese corpus is clause decisions made with paragraph context, not sentence selection.** The clean evidence chain is:

1. the official benchmark scores four classifications but the released data also contains `promise_string` and `evidence_string` on every applicable row;
2. the annotations are geometrically clause-like: sentence boundaries align poorly and impose a char-F1 ceiling of 0.5414, whereas comma-delimited clauses raise the ceiling to 0.9748 (`span_extraction_feasibility.md`);
3. under the same backbone and training protocol, changing only sentence units to clause units raises three-seed mean char-F1 from 0.4857 to 0.5889 for promises and from 0.6356 to 0.6663 for evidence (`tagger_ablation_*.md`);
4. removing paragraph context loses 0.0590 and 0.0238, while changing RoBERTa to MacBERT moves only -0.0066 and +0.0013;
5. char-F1 alone is unsafe because copying the whole paragraph scores 0.6004 on evidence but only 0.4% exact match, so char-F1, IoU, and exact match must be reported together.

That is a research question, a method, controlled experiments, an ablation, and an evaluation argument. It is not a leaderboard report.

The guideline-diagnosis story is interesting but not yet a paper-grade causal claim. The 147 disputed rows have not been adjudicated; the strongest criterion feature family was designed after inspecting test disagreements; the abstention result is accuracy-only and hurts macro-F1; and the cross-lingual prediction has not been tested. Those facts do not make the work worthless. They make it a second paper after the missing validation is done.

## The story the paper should tell

The single claim should be:

> In Traditional Chinese ESG paragraphs, promise and evidence spans are best modelled as clause selections whose representations retain paragraph context; sentence selection loses the boundary structure visible in the annotations.

The reader should meet the evidence in this order:

1. Why classification labels are insufficient for audit use, and which two released fields make localisation possible.
2. What the gold spans look like: discontinuity, within-sentence placement, boundary alignment, and the sentence/clause ceilings.
3. The clause tagger and the controlled sentence baseline.
4. Main results on char-F1, IoU, and exact match.
5. Ablations for decision unit, context, and encoder.
6. Metric pathology (whole-paragraph copying) and limitations.

The paper should not open with `evidence_quality`, four commercial APIs, or the 147 disagreements. None is needed to understand or validate the span contribution.

## What is worth doing next, ranked by value gained per cost

### 1. Repair the gold alignment and rerun every span score

This is mandatory, not optional polishing.

- **Comparison:** current first-substring `gold_mask` versus a validated alignment that accounts for every cleaned span piece, with an explicit `unalignable` state rather than silently dropping text.
- **Rows:** all 1,625 promise-positive and all 1,335 evidence-positive test rows; threshold selection remains on the corresponding 1,627 and 1,345 train+validation rows.
- **Metrics:** alignment coverage by row and piece; then char-F1, IoU, and exact match on (a) all rows with unalignable rows counted as failures and (b) the fully alignable subset.
- **Outcome A — clause gains survive:** the paper may claim the result is not an artefact of silently empty gold masks.
- **Outcome B — gains shrink but remain:** claim a smaller, properly bounded advantage and make alignability part of the task definition.
- **Outcome C — ranking changes:** withdraw the present performance claim; retain only the annotation-geometry finding until retraining.
- **Cost:** CPU if the cached probabilities are recovered; otherwise one GPU to reproduce the five-fold runs. No API. No human annotator, although manual review of unresolved alignments would improve it.

Why this is first: `gold_mask()` silently ignores every piece not found in the paragraph. My direct check of the released test file found 120/1,625 promise rows and 125/1,335 evidence rows with at least one unmatched cleaned piece; 114 and 113 respectively become entirely empty gold masks. Those counts are not in an existing report, which is itself evidence that the current reporting missed the problem. Exact match can be inflated whenever a model also predicts nothing on those rows.

### 2. Add a same-backbone token-level BIO baseline, with the same three seeds

- **Comparison:** clause tagger versus token-level BIO (preferably BIO+CRF as a second row), both using `hfl/chinese-roberta-wwm-ext-large`, the same train/validation/test rows, folds, maximum length, optimiser, and threshold/model-selection policy.
- **Rows:** all alignable applicable rows after experiment 1; report the all-row score separately.
- **Metrics:** char-F1, IoU, exact match, invalid-transition rate, and runtime.
- **Outcome A — clause wins:** licenses the strong claim that the clause inductive bias beats an unconstrained boundary model, not merely a weak sentence reduction.
- **Outcome B — tie:** licenses an efficiency/simplicity claim, not an accuracy-superiority claim.
- **Outcome C — BIO wins:** the paper should become an evaluation-and-benchmark paper; clause geometry remains useful, but “clause is the right model” must be withdrawn.
- **Cost:** one GPU; no API; no human annotator.

This is the most obvious reviewer demand. The present baseline proves only “clause beats sentence.” A token-level model is the natural alternative and is absent.

### 3. Repeat every controlled variant under the same seeds and use paired inference

- **Comparison:** full clause model, sentence units, no paragraph context, and MacBERT, each under the same three seeds and folds.
- **Rows:** the same rows as experiment 2.
- **Metric:** per-row char-F1 and IoU, with a paired bootstrap nested within seed or a seed-by-row mixed analysis; also report the distribution of deltas across seeds.
- **Outcome A — unit and context effects remain in every seed:** licenses robust claims about both design choices.
- **Outcome B — only unit survives:** keep clause granularity as the contribution and demote paragraph context.
- **Outcome C — either effect changes sign:** report seed sensitivity and stop using two standard deviations from the full model as a significance surrogate.
- **Cost:** one GPU, roughly several repetitions of the existing five-fold jobs; no API; no human annotator.

At present only the full model has three seeds. The sentence/context/backbone variants are single runs. The reported standard deviation is a useful noise scale but not a paired test of the ablations.

### 4. Run a blinded human acceptability evaluation of extracted spans

- **Comparison:** gold span, sentence baseline, clause model, and token baseline from experiment 2, with system identity hidden.
- **Rows:** 150 test paragraphs stratified by target field, single- versus multi-piece gold, table/list versus prose, and exact-match success versus failure.
- **Metric:** proportion judged “sufficient to verify the label without reopening the paragraph,” boundary acceptability, pairwise preference, and inter-annotator agreement.
- **Outcome A — clause output is preferred despite low exact match:** licenses the practical audit-assistance claim.
- **Outcome B — gold and multiple systems are often equally acceptable:** shows exact match is too strict and supports an acceptability metric.
- **Outcome C — outputs are usually unusable:** limits the contribution to benchmark construction and removes deployment language.
- **Cost:** no GPU after predictions exist; no API; two human annotators plus an adjudicator. Domain expertise is useful but not strictly necessary for boundary acceptability.

This would turn the low 14.1%/12.3% exact-match figures from an embarrassment into a research question about valid alternative spans.

### 5. Cross-lingual span replication on ML-Promise, but be honest about what is available

- **Comparison:** sentence, clause, and token-level extraction on ML-Promise Japanese; then Chinese only after paragraph text is re-extracted and aligned from the supplied PDFs.
- **Rows:** every Japanese row in `research/data/raw/mlpromise/` with non-empty `data` and a non-empty, fully alignable target span; for Chinese, every one of the 899 released annotations whose paragraph can be reconstructed and aligned. Report the post-validation counts rather than assuming all rows are usable.
- **Metrics:** the same char-F1, IoU, exact match, and alignment coverage as VeriPromiseESG4K.
- **Outcome A — clause wins in Japanese and reconstructed Chinese:** licenses a CJK-level generalisation claim.
- **Outcome B — clause wins only in Traditional Chinese:** the result is corpus- or punctuation-convention-specific.
- **Outcome C — token/sentence models win elsewhere:** keep the VeriPromiseESG4K finding local and explain the boundary distributions by language.
- **Cost:** GPU plus substantial data engineering; no API; a Japanese reader would help audit segmentation, and Chinese PDF reconstruction may need human spot checks.

The repository does **not** permit a clean five-language span experiment today. Japanese is the only language with both paragraph text and spans. Chinese has spans but no paragraph text; English and French have text but no spans; Korean has neither. Saying “ML-Promise gives five-language span validation” would be false (`mlpromise_inventory.md`).

### 6. Adjudicate the 147 unanimous human–model disagreements if the guideline paper is revived

- **Comparison:** released gold label versus the four-model unanimous label against an independent adjudicated label.
- **Rows:** all 147 disputed rows, mixed blind with 147 population-matched unanimous-and-gold control rows.
- **Metric:** accuracy of gold and model consensus against adjudication; `Clear`/`Not Clear` confusion; annotator agreement; preference conditional on numbers, hedge wording, and promise--evidence link.
- **Outcome A — adjudicators mostly side with gold:** licenses “models apply a different criterion.”
- **Outcome B — adjudicators mostly side with models:** licenses a dataset-quality/documentation finding, not a model-failure finding.
- **Outcome C — adjudicators disagree heavily:** licenses an ambiguity/plural-judgment paper and invalidates a single-gold framing.
- **Cost:** no GPU; no API because predictions are cached; two independent ESG-aware human annotators and a tie-breaking adjudicator.

This experiment has enormous scientific value but a worse value/cost ratio for the **span paper** because it belongs to a different claim. For the diagnostic paper it would be rank 1.

### 7. Test the documentation hypothesis cross-lingually on ML-Promise

- **Comparison:** guideline-verbatim features versus an extended operationalisation, fitted separately to `Clear` versus `Not Clear` in English, French, and Japanese; Chinese can enter after text reconstruction.
- **Rows:** all rows with actionable evidence and a valid clarity label in each usable language.
- **Metric:** five-fold out-of-fold chance-corrected balanced accuracy and the extended-minus-verbatim gain, exactly paralleling script 39.
- **Outcome A — the gain is largest where published guidance is least operational:** supports the documentation-transmission hypothesis beyond one corpus.
- **Outcome B — no relationship:** the VeriPromiseESG4K result is dataset-specific and should not be generalised.
- **Outcome C — detailed ML-Promise guidance predicts labels well:** strengthens a narrower claim that releasing operational detail matters.
- **Cost:** CPU for the classifiers; no GPU; no API should be necessary. It does require bilingual humans to construct faithful, non-post-hoc feature sets. An API-generated lexicon would weaken the experiment.

This is the right cross-lingual test of the *diagnostic* claim. ML-Promise Appendix B is especially important because it explicitly names timelines, figures, percentages, involved parties, and intended outcomes. That is a natural positive control against VeriPromiseESG4K's shorter public field definition.

### 8. Do not run more models on `Misleading`

The class has three test instances. Another API sweep cannot turn that into a stable estimate. Make `Clear` versus `Not Clear` the primary evidence-quality analysis, report `Misleading` descriptively with its support, and provide a sensitivity table showing how the headline changes when that class is excluded. No GPU, API, or annotator is needed. The currently reported “frontier models find 0/3” is anecdotal; it should not anchor any contribution.

## Results that should be cut, not extended

- **Cut the heterogeneous ensemble result.** Accuracy moves to 0.8300, but always predicting `Clear` reaches 0.8285, while balanced accuracy and discrimination fall. It is a metric caution, not a method.
- **Cut the abstention section from this paper.** The feature set was selected after looking at test disagreements, its positive result is accuracy-only, and criterion-only abstention loses to random on macro-F1. It needs a clean held-out feature-design protocol before becoming a paper.
- **Cut the six-input locate-then-judge table.** The positive control fails in 33 of 40 bootstrap seeds. “No detected effect under an insensitive design” is not worth scarce pages.
- **Cut the ensemble-diversity discussion.** Two measurement points do not support a new ensemble claim, and one “positive” point is majority-class drift.
- **Cut commercial-model reason mining from the span paper.** The response-format negative result is real, but it is orthogonal. Preserve it for a short methods note or the later guideline paper.
- **Cut the chance-corrected four-field difficulty analysis from the span paper.** It is careful and correct, but it answers why `evidence_quality` is hard, not how to extract spans.

## Were any section 7 claims withdrawn too aggressively?

No important claim was withdrawn too aggressively.

- **7.1 substitution mechanism:** correctly withdrawn. Once verdict is held fixed, the lift is +0.034 and changes sign by model; the detector also fires on prompt wording; decomposition changes the co-occurrence. The remaining observation is about generated text format, not internal mechanism.
- **7.2 “no released document mentions numbers”:** correctly withdrawn, and the correction should go further. `DATA.md` names data under `evidence_status`, while ML-Promise Appendix B explicitly names quantitative and qualitative clarity cues. Any argument must distinguish the VeriPromiseESG4K public field definition from the predecessor's fuller annotation guidance.
- **7.3 every criterion fits models better:** correctly withdrawn. The paired same-row result for family B is exactly +0.000.
- **7.4 task-formalisation exclusion:** correctly withdrawn as an exclusion. It remains permissible to state the raw null as an inconclusive experiment, but it has no place in a seven-page span paper.
- **7.5 timeline floor effect:** correctly withdrawn. A tool without a demonstrated positive control cannot turn a zero into evidence of no divergence.
- **7.6 heterogeneous ensemble:** correctly demoted; I would cut it entirely.
- **7.7 criterion features beat confidence:** correctly withdrawn. The combined signal's +0.0206 over confidence is real under the reported split, but test-informed feature design prevents a strong general claim.

The one result worth preserving from this cluster is the response-format negative result: asking several criteria in one sentence can manufacture co-occurrence. It is valid, but it belongs in a different paper.

## Number and source audit

Numbers I checked and found consistent with their named reports:

- sentence-to-clause char-F1: 0.4857 to 0.5889 (promise), 0.6356 to 0.6663 (evidence);
- no-context deltas: -0.0590 and -0.0238;
- MacBERT deltas: -0.0066 and +0.0013;
- sentence/clause ceilings: 0.5414 and 0.9748;
- whole-paragraph evidence: char-F1 0.6004, exact match 0.4%;
- applicable test rows: 1,625 promise and 1,335 evidence;
- three-seed standard deviations: 0.0033 and 0.0010;
- the current draft's 49 unique reports/tickers agrees with the released data after normalising the Wistron name variant.

Mismatches or dangerous near-mismatches:

1. **50 companies versus 49.** The dataset paper repeatedly says 50 companies. The released data has 49 unique tickers; ticker 3231 appears under two name variants and produces a naive count of 50. The paper must state which source it is reporting, not silently choose one.
2. **ML-Promise Chinese size.** The published prose says 610 Chinese instances, its Table 1 exposes 199 for promise identification, and the released files contain 899. Do not write a single Chinese sample size without identifying paper/table/release.
3. **0.6189 versus 0.6194 without `Misleading`.** `subtask_difficulty.md` reports 0.6189 on the all-test-row label-average convention; `official_test_scoring.md` reports 0.6194 after removing the class from both the data and averaging. They are different row sets/operations, not interchangeable roundings.
4. **Gemini macro-F1 0.4143 versus 0.4165.** `frontier_api_ceiling.md` contains the earlier 0.4143, while the strict convention that excludes 80 failed calls reports 0.4165 on 1,255 rows. Only the latter matches section 8's standing convention.
5. **Single-seed versus three-seed tagger scores.** Script 32 reports 0.5903/0.6666 for one seed; script 42 reports 0.5889/0.6663 as three-seed means. The new draft uses the latter consistently.
6. **Raw span coverage versus evaluable coverage.** The fields are non-empty on every applicable row, but this is not the same as every span being found in `data`. The evaluation silently drops unmatched pieces. This is the most serious unchecked assumption in the span results.

## What the VeriPromiseESG4K paper changes

It strengthens the span paper in three ways:

- It says all official labels and both span fields were manually annotated; GPT-5 Pro was used only to build the candidate pool, not labels or boundaries.
- It explicitly justifies paragraph-level units because promises and evidence can be distributed across sentences. That is direct external support for retaining paragraph context while making local clause decisions.
- Its reference baseline evaluates only the four classification tasks. It therefore leaves the span benchmark genuinely open.

It strengthens the diagnostic reading more cautiously:

- It calls evidence quality the lowest-agreement field at alpha 0.940 and reports that disagreement arose when text used concrete-sounding words without measurable outcomes.
- It says guidelines were repeatedly refined in weekly meetings and final disputes were adjudicated by a sustainability expert. The released short `DATA.md` is therefore not the whole decision process.

It also creates problems:

- It says 50 companies where the release resolves to 49.
- It does **not** say that no published system has ever used the spans. It shows that its own shared-task baseline does not. The paper should write “we found no published evaluation on these fields” and describe the search scope, not assert universal absence.
- High alpha does not rule out systematic disagreement on the 147 selected rows. Alpha is an aggregate over the annotation process; it cannot substitute for adjudicating that subset.

I found no statement in the dataset paper that contradicts the clause hypothesis. Its paragraph-level motivation supports it. The main contradiction is documentary: nominal company count, plus the gap between a rich internal refinement/adjudication process and the short public field definitions.

## Venue-bar assessment

The 30 August omnibus draft does **not** clear the “not merely a competition system report” bar reliably. It has too many stories, and several sections exist mainly because experiments were run rather than because the central claim needs them.

The new span-centred draft has the right shape and, after the mandatory alignment repair and a token-level baseline, should clear the bar comfortably. It uses the shared-task data as material, defines a discipline-standard extraction problem and metrics, makes a controlled comparison, and provides ablations and a falsifiable generalisation hypothesis. That is exactly the contribution shape established by the NTCIR-18 special-session papers.

As it stands today, before those two fixes, my recommendation would be **borderline reject**. The missing unconstrained baseline is obvious, and the silent empty-gold alignment bug can contaminate exact match and mean overlap scores. Fix those, repeat the controlled variants across seeds, and the paper becomes a credible accept rather than a well-written system report.
