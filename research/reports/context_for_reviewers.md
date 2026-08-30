# Context for a reviewer seeing this project for the first time

This file exists so someone can be dropped into the repository and understand,
in one read, what was done, what survived scrutiny, what was withdrawn, and
where every number lives. It is written for a reader with no prior exposure to
the conversation that produced the work.

---

## 1. The venue and what it demands

Target: **NTCIR-19, AI CUP VeriPromiseESG special session**, Tokyo NII,
8–10 December 2026.

The organisers' stated review criterion, verbatim from the call:

> 本次研討會論文的審查重點在於整體的「學術性與研究價值」。投稿內容不能僅是一份
> 單純的「競賽系統報告」，而需具備完整的研究動機、方法論述與實驗探討。

So a leaderboard write-up is explicitly out of scope. The paper has to have a
research question, a method, and experiments that bear on it.

Practical constraints: ~7 pages, ACM two-column format (the current draft is
single-column A4 and has **not** been converted). Proceedings are in English;
the current draft is in Traditional Chinese by the author's choice and would
need translation.

**Precedent.** The three NTCIR-18 AI CUP special-session papers all took the
same shape: use the competition dataset as *material*, then define
discipline-standard metrics for the paper's own question. None reported a
leaderboard score as evidence.

| paper | metrics it defined for itself |
|---|---|
| KAP: MLLM-assisted OCR Text Enhancement | MRR, AP@1, plus a standalone "Evaluation Methodology" section and ablations |
| Ensemble-Based Multi-Specialty Retrieval | Accuracy, F1, Precision; unsupervised and supervised ensemble contrasts |
| Translation or Multilingual Retrieval? | MRR@10, NDCG@10, Recall@100; explicit baseline, approximate upper bound, paired t-test, error analysis, runtime |

The third is the closest template: a controlled comparison that proposes no new
model.

---

## 2. The task and the data

**VeriPromiseESG4K.** 4,000 paragraphs from the 2024 sustainability reports of
49 Taiwanese listed companies. Split 1,000 train / 1,000 validation / 2,000
test. The full field definitions the organisers released are in `data/DATA.md` —
**read that file, it is short and it is the object of most of this project's
analysis.**

Four classification fields and two span fields:

| | field | question | classes |
|---|---|---|---|
| ① | `promise_status` | is there an ESG promise? | 2 |
| ② | `evidence_status` | is there supporting evidence? | 3 |
| ③ | `evidence_quality` | how good is the evidence? | 4 |
| ④ | `verification_timeline` | when can it be checked? | 5 |
| | `promise_string` | which sentence is the promise (span) | — |
| | `evidence_string` | which sentence is the evidence (span) | — |

The two span fields have **100% coverage on positive rows and have never been
used by any published system** — the shared task only scores the four
classification fields. That is the opening for section 5 of the draft.

Inter-annotator agreement reported by the dataset publishers (Krippendorff's
alpha): ① 0.977, ② 0.962, ③ 0.940, ④ 0.955.

---

## 3. What is and is not in this repository

**Present:**

| what | where |
|---|---|
| The dataset | `data/vpesg4k_*.csv` / `.json` |
| The organisers' field definitions | `data/DATA.md` |
| The submission guidelines | `data/AI_CUP_2026_VeriPromiseESG_Submission_Guidelines.pdf` |
| ML-Promise published tables, hand-transcribed with citation | `research/data/mlpromise_reference.json` |
| ML-Promise raw data, five languages | `research/data/raw/mlpromise/` |
| The competition system's outputs | `submissions/*.csv` |
| 52 experiment scripts | `research/scripts/` |
| One report per script | `research/output/` |
| Two long progress reports | `research/reports/progress_report_2026-08-2{5,7}.*` |
| The span/divergence report | `research/reports/span_and_divergence_2026-08-28.tex` |
| **The current paper draft (Chinese)** | `research/reports/paper_draft_zh_2026-08-30.tex` |
| A structural checker for that draft | `research/reports/check_tex.py` |

**The references, all in `research/refs/`:**

| file | what it is | why you want it |
|---|---|---|
| `VeriPromiseESG4K.pdf` | Day et al., the dataset paper for this shared task (5 pp) | The alphas, the annotation process, the retrieval baseline this project quotes (③ 0.3549 → 0.3599, ④ 0.3090 → 0.4227), and the phrase "concrete-sounding words but lacked measurable outcomes" |
| `2025.emnlp-main.1028.pdf` | Seki et al., ML-Promise, EMNLP 2025 pp. 20353–20366 (14 pp) | The multilingual predecessor dataset. Same four-field formalisation, five languages, 610 Chinese instances |
| `2023.finnlp-1.11.pdf` | Chen et al., Multi-Lingual ESG Issue Identification (ML-ESG) | The shared-task lineage these organisers come from |
| `2023.finnlp-2.6.pdf` | ML-ESG, FinNLP-2023 | same lineage |
| `2024.finnlp-1.22.pdf` | ML-ESG, FinNLP/KDF/ECONLP 2024 (9 pp) | same lineage, most recent |
| `01-AICUP-AICUP-HsuH.pdf` | KAP: MLLM-assisted OCR Text Enhancement (7 pp) | NTCIR-18 special session, **the format precedent** |
| `02-AICUP-AICUP-ChaoC.pdf` | Ensemble-Based Multi-Specialty Retrieval (7 pp) | NTCIR-18 special session |
| `03-AICUP-AICUP-ChiuY.pdf` | Translation or Multilingual Retrieval? (7 pp) | NTCIR-18 special session. **The closest template**: a controlled comparison proposing no new model, with baseline, approximate upper bound, paired t-test, error analysis and runtime |
| `chuang25b.pdf` | Chuang et al., Learning to Route LLMs with Confidence Tokens (Self-REF), ICML 2025 | The method Appendix B reimplements |
| `NeurIPS-2024-simpo-...pdf` | Meng et al., SimPO | Considered and rejected as a fix; see section 7 of the progress reports |
| `TEAM_10671_competition_report_zh.pdf` | The authors' own competition report (15 pp, Chinese) | Where the system in section 4 came from, and the training details |

**One warning about that last file.** It is a competition report and it opens
with a leaderboard placement. **The paper must not mention the ranking, the
leaderboard, the competition's scoring weights, or that `evidence_quality`
carries the largest weight in that formula.** This is a standing instruction
from the author and it has been given more than once. Use the report for
reproducible detail — hardware, hyperparameters, what was trained — and nothing
else.

Also absent: nothing of consequence. Anything cited in the current draft that is
not in `research/refs/` is cited from general knowledge and should be checked
before it goes into a submission.

---

## 4. The competition system (the material, not the contribution)

Three Chinese encoders — `chinese-roberta-wwm-ext-large`,
`chinese-macbert-large`, `Erlangshen-DeBERTa-v2-320M`, roughly 300M parameters
each — each run through 5-fold cross-validation, giving 15 fine-tuned models
whose class probabilities are averaged (soft voting). Outputs in
`submissions/`. Two variants differ only in whether decision thresholds were
tuned; `ens_3way_raw` is the untuned one and it is **better** on joint exact
match (0.5130 vs 0.4880).

Most of the diagnosis section instead uses a **single** encoder
(`chinese-roberta-wwm-ext-large`, condition A of script 36), fine-tuned on the
1,345 train+val rows that have evidence, five folds. That is the system the
frontier models are compared against.

---

## 5. The research question

On a scale where the four fields are comparable — balanced accuracy corrected
for the number of classes, `(BA − 1/K)/(1 − 1/K)`, so 0 is chance — the system
scores:

| ① | ② | ③ | ④ |
|---:|---:|---:|---:|
| +0.612 | +0.536 | **+0.289** | +0.594 |

③ is less than half of any other field. The failure is concentrated in one
class: `Not Clear`, 226 test rows, F1 0.3556, against 0.57–0.70 for every other
field's minority class.

**Do not motivate this with macro-F1.** ③'s macro-F1 of 0.4642 against ④'s
0.6545 is 0.155 an artefact of `Misleading`, which has 3 test rows and scores
0.0000. Drop that class and ③ is 0.6189. Script 51 makes this explicit.

The question the paper asks: **why is this one field hard, and is it hard in the
same way as the others?**

---

## 6. What the paper currently argues

**Thesis: the guideline gives checkable surface cues for three fields and three
adjectives for this one. Words did not transmit the criterion; labelled examples
did.**

Evidence, each with the script that produces it:

| claim | number | script |
|---|---|---|
| Official verbatim cues reproduce the gold label | ③ **+0.002**, vs ① +0.298, ② +0.375, ④ +0.210 | 39 |
| Adding a workable operationalisation is worth most on ③ | ③ **+0.375**, others −0.003 to +0.132 | 39 |
| The guideline's only worked example (「持續推進」「努力改善」) occurs | **0 times in 742 paragraphs** | 34 |
| Of three candidate criteria, the one operationalising the guideline's own illustrated rule is worst at reproducing the labels | A 0.592 [0.561, 0.623], below B's and C's interval floors | 34 |
| Four frontier models all lose to a 330M fine-tuned encoder | 0.354–0.449 vs 0.4655 | 26, 45, 47 |
| Labelled examples do transmit it | same Qwen3-8B, macro-F1 0.3914 → 0.4551 | 44 |

Positive results:

| result | number | script |
|---|---|---|
| Clause-unit span tagger, against the sentence-unit supervised baseline | char-F1 0.4857 → **0.5889** (promise), 0.6356 → **0.6663** (evidence), 3 seeds | 32, 42 |
| Abstention: criterion features + model confidence, vs random abstention at 80% coverage | **+0.0770 [+0.0562, +0.0955]** accuracy | 38 |
| Whole-row abstention, criterion features + a zero-cost structural check | **+0.0445 [+0.0244, +0.0625]** joint exact match | 46 |

A methodological negative result, which the draft treats as a contribution:

> Asking a model to answer several criteria in **one sentence** and then counting
> how often two criteria co-occur in that sentence **manufactures the
> co-occurrence**. Same models, same paragraphs: split the four criteria into
> four separate questions and the rate falls from 0.71–0.75 to 0.04–0.32.
> (Script 40, sub-answers cached in `research/artifacts/s40/*__decomp.json`.)

---

## 7. Claims that were withdrawn, and why. Do not resurrect these.

This project ran an adversarial review of its own draft and killed several of
its own results. A reviewer who does not know this will suggest putting them
back.

**7.1 The substitution mechanism.** The draft used to argue that models comply
with the stated "not vague" criterion in form but instantiate it as "contains no
numbers", evidenced by a co-occurrence of 0.837. Withdrawn for three reasons:

1. **Confounded by the model's own verdict.** Holding the verdict fixed, reasons
   that cite vagueness name missing figures 0.850 of the time and reasons that do
   not cite vagueness name them 0.816 of the time — a lift of +0.034 whose sign
   flips across models. The 0.837 is P(figures | Not Clear) = 0.846 with the
   vagueness code contributing nothing.
2. **The vagueness detector fires on our own prompt.** The `Not Clear` option
   gloss written into scripts 26 and 35 contains 「流於表面」, which is in neither
   `DATA.md` nor the guidelines PDF. That phrase triggers 267 of 515 vagueness
   codes; 表面 triggers 304. The word 模糊 that the guideline actually uses fires
   **once in 1,343 reasons**.
3. **It is a response-format artefact.** See section 6's negative result.

**7.2 "No released document mentions numbers."** False. `data/DATA.md` line 22,
the definition of the adjacent field `evidence_status`, gives 「數據」 as the
first example of what evidence is. The draft now describes this as criterion
bleed between adjacent fields, and family C is renamed accordingly — it is also
not purely numeric, since two of its seven patterns match named bodies and
`ISO\s?\d+` / 認證 / 查證.

**7.3 "Every criterion describes the models better than the annotators."** The
gold column used 1,332 rows and the consensus column 738 rows selected for
unanimity. Scoring both targets on the same 735 rows, the official link
criterion's gap is **+0.000**. Only families A (+0.074) and C (+0.085) survive.

**7.4 Test 3 as an exclusion.** Six input conditions (whole paragraph vs
annotated span vs predicted span vs random span) showed no effect, which was
read as excluding the task-formalisation explanation. But the positive
control — replacing the input with random text of the same length — excludes
zero on only **7 of 40 bootstrap seeds**, mean interval [−0.0392, +0.0003].
The design has no demonstrated sensitivity, so its null is uninformative rather
than negative. Script 52.

**7.5 Subtask ④'s zero divergence as a floor effect.** Also wrong, in the other
direction: chance-corrected, ④'s two fits are both +0.213, *above* ①'s +0.182.
There is no floor. The real problem is that the divergence instrument has never
been shown able to detect anything on a 4-class timeline field.

**7.6 The heterogeneous ensemble as a positive result.** Averaging the
fine-tuned transformer with a zero-shot Qwen3-8B raises accuracy 0.8052 →
0.8300, interval excluding zero. But **always predicting `Clear` scores 0.8285**
on these rows — the gain is two rows out of 1,335 — while balanced accuracy
falls 0.4871 → 0.4806 and discrimination falls +0.461 → +0.442. Demoted to a
demonstration that accuracy cannot be read alone under this class imbalance.

**7.7 "Criterion features beat the model's own confidence at predicting its
errors."** Paired bootstrap: +0.0066 [−0.0112, +0.0243]. Not detected. What
survives is that the two **combined** beat confidence alone, +0.0206 [+0.0047,
+0.0318]. Note also that on macro-F1 the criterion signal **loses to random
abstention** at every coverage from 80% down; the abstention claim is
accuracy-only and the draft says so.

---

## 8. Conventions that every script follows

Two decisions, applied at source in every script (`research/scripts/47_...` is
the one deliberate exception, since it exists to show what the lenient
convention is worth):

1. **A failed API call is excluded, never scored.** `gemini-3.7-flash` returned
   HTTP 503 on 80 of the 1,335 evidence rows. Reading an empty response as
   `Clear` hands a correct answer to the least reliable model on a set that is
   82.85% `Clear`. Its strict score is 0.4165 / 0.7124 on 1,255 rows.
2. **A predicted `N/A` on a row that has evidence is wrong**, not absent.

`research/scripts/probe_groups.py` is the **single** definition of the two
paragraph groups used by scripts 34, 40, 43, 48, 49 and 50: 338 probed rows,
147 disputed (four models unanimous and differing from the annotators) and 191
control (unanimous and matching).

---

## 9. The largest open gap

**Nobody has adjudicated the 147 disputed rows.** Script 30 built a 30-row
blind adjudication form, population-proportional, and it was never run. Until
someone independent decides who is right on those rows, two readings are not
separated:

- the models are applying a different criterion than the annotators, or
- the annotators are inconsistent on exactly those rows.

Script 31 mildly favours the second: on disputed rows the neighbour-majority
label matches gold 80% of the time against 87% on control.

The draft names this as its first limitation.
