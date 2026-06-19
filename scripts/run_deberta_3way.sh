#!/usr/bin/env bash
# OPTIONAL last-day gamble: add DeBERTa-v2 as a 3rd backbone.
# Trains DeBERTa CV5 (fp32, AMP is incompatible), builds roberta+macbert+deberta
# ensemble, re-tunes thresholds, generates a 3-way CSV, and prints both OOF scores
# so you can decide. Run in tmux:  bash scripts/run_deberta_3way.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

ROB=experiments/20260617_080052_roberta_large_mergeval_cv5
MAC=experiments/20260617_191631_macbert_large_mergeval_cv5

echo "### Training DeBERTa-v2-320M CV5 (fp32, bs8, lr2e-5)  $(date)"
$PY scripts/train_baseline.py \
  --experiment-name deberta_v2_320m_mergeval_cv5 \
  --model-name IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese \
  --merge-val --cv-folds 5 --cv-stratify-field evidence_status \
  --max-len 384 --batch-size 8 --epochs 15 --class-weight balanced \
  --lr 2e-5 --seed 42 --tensorboard

DEB=$(ls -td experiments/*deberta_v2_320m_mergeval_cv5 | head -1)
echo "### DeBERTa dir: $DEB"

echo "### Tuning 3-way ensemble OOF  $(date)"
$PY scripts/tune_thresholds.py \
  --cv-dir "$ROB" "$MAC" "$DEB" \
  --output experiments/ens_3way_bias.json

echo "### Generating 3-way submission CSV  $(date)"
$PY scripts/predict_test.py \
  --checkpoint $ROB/fold{1,2,3,4,5}/best_model.pt \
               $MAC/fold{1,2,3,4,5}/best_model.pt \
               $DEB/fold{1,2,3,4,5}/best_model.pt \
  --bias-file experiments/ens_3way_bias.json \
  --output submissions/ens_3way_tuned.csv

echo "========================================================================"
echo "DECISION: compare OOF scores"
echo "  roberta+macbert (current best) OOF = 0.63660  -> submissions/ens_rob_mac_tuned.csv"
THREE=$($PY -c "import json;print(json.load(open('experiments/ens_3way_bias.json'))['summary']['weighted_after'])")
echo "  roberta+macbert+deberta        OOF = $THREE  -> submissions/ens_3way_tuned.csv"
echo "Submit the 3-way ONLY if its OOF > 0.63660. Otherwise submit ens_rob_mac_tuned.csv."
echo "========================================================================"
