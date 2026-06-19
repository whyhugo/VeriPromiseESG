#!/usr/bin/env bash
# Generate the FINAL submission: roberta+macbert 5-fold ensemble + tuned biases.
# OOF weighted_score = 0.6366 (validated to transfer to test within <0.6%).
# Run in tmux:  bash scripts/make_final_submission.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

ROB=experiments/20260617_080052_roberta_large_mergeval_cv5
MAC=experiments/20260617_191631_macbert_large_mergeval_cv5
BIAS=experiments/ens_rob_mac_bias.json
OUT=submissions/ens_rob_mac_tuned.csv

# Optional cleanup of the killed partial retrain.
rm -rf experiments/20260618_060345_macbert_large_mergeval_cv5

$PY scripts/predict_test.py \
  --checkpoint $ROB/fold{1,2,3,4,5}/best_model.pt $MAC/fold{1,2,3,4,5}/best_model.pt \
  --bias-file $BIAS \
  --output $OUT

echo "=== validate ==="
rows=$(wc -l < "$OUT")
echo "rows=$rows (expect 2001)"
head -2 "$OUT"
[ "$rows" -eq 2001 ] && echo "OK -> submit $OUT  (expected LB ~0.63)" || echo "ROW COUNT WRONG, do not submit"
