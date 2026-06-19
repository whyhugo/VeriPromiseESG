#!/usr/bin/env bash
# Last-day ensemble-diversity runs. Sequential (one large model fits 24GB at a time).
# Each CV5 run is ~1h45m. Launch with:  nohup bash scripts/run_lastday.sh > lastday.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

COMMON="--merge-val --cv-folds 5 --cv-stratify-field evidence_status \
  --max-len 384 --batch-size 16 --epochs 15 --class-weight balanced --fp16 \
  --seed 42 --tensorboard"

echo "### [1/2] macbert-large CV5  $(date)"
$PY scripts/train_baseline.py \
  --experiment-name macbert_large_mergeval_cv5 \
  --model-name hfl/chinese-macbert-large \
  --lr 4e-5 $COMMON

echo "### [2/2] DeBERTa-v2-320M CV5  $(date)"
# DeBERTa-v2 is incompatible with AMP fp16 ("unscale FP16 gradients"): run fp32.
# Also lr-sensitive (use 2e-5) and heavier without fp16, so drop batch to 8.
$PY scripts/train_baseline.py \
  --experiment-name deberta_v2_320m_mergeval_cv5 \
  --model-name IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese \
  --merge-val --cv-folds 5 --cv-stratify-field evidence_status \
  --max-len 384 --batch-size 8 --epochs 15 --class-weight balanced \
  --lr 2e-5 --seed 42 --tensorboard

echo "### ALL DONE  $(date)"
