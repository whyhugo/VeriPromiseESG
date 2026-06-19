#!/usr/bin/env python3
"""Tune per-class decision biases on out-of-fold (OOF) predictions.

Reuses the 5 fold checkpoints from a `--cv-folds 5` run. For each fold it loads
that fold's best_model.pt and predicts the fold's *validation* indices, which
reconstructs leak-free OOF softmax probabilities over all labeled samples (the
same data the CV score was computed on). It then coordinate-ascends a per-class
additive log-prior bias to maximize macro-F1 per field.

Because the OOF setup mirrors the CV/test generalization gap (each sample is
scored by a model that never trained on it), biases tuned here transfer to the
test set. Apply them at inference with `predict_test.py --bias-file`.

Example
-------
python3 scripts/tune_thresholds.py \
    --cv-dir experiments/20260617_080052_roberta_large_mergeval_cv5 \
    --seed 42 --cv-stratify-field evidence_status \
    --output experiments/.../class_bias.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from predict_test import EVAL_FIELDS, MultiTaskClassifier, TestDataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune per-class biases on OOF predictions.")
    p.add_argument("--cv-dir", type=Path, nargs="+", required=True,
                   help="One or more --cv-folds run dirs (fold1..foldN). Multiple dirs are "
                   "soft-voted as an ensemble; they must share the same seed/stratify so folds align.")
    p.add_argument("--train-path", type=Path, default=Path("data/vpesg_4k_train_1000.json"))
    p.add_argument("--val-path", type=Path, default=Path("data/vpesg4k_val_1000.json"))
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-stratify-field", default="evidence_status", choices=list(EVAL_FIELDS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=Path, default=None, help="Where to write class_bias.json (default <cv-dir>/class_bias.json).")
    return p.parse_args()


def norm_label(value: Any, labels: list[str]) -> str:
    if value in ("", None) and "N/A" in labels:
        return "N/A"
    return value


def load_merged(train_path: Path, val_path: Path) -> list[dict[str, Any]]:
    with train_path.open(encoding="utf-8") as f:
        data = json.load(f)
    with val_path.open(encoding="utf-8") as f:
        data += json.load(f)
    return data


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_merged(args.train_path, args.val_path)
    n = len(data)
    num_labels = {f: len(l) for f, l in EVAL_FIELDS.items()}

    # Reproduce the exact CV split used during training.
    strat = [norm_label(x.get(args.cv_stratify_field, ""), EVAL_FIELDS[args.cv_stratify_field]) for x in data]
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    # OOF softmax probabilities per field, summed over backbones (then averaged).
    oof_prob = {f: torch.zeros(n, num_labels[f]) for f in EVAL_FIELDS}

    for cv_dir in args.cv_dir:
        print(f"== backbone dir: {cv_dir} ==")
        for fold, (_, val_idx) in enumerate(skf.split(data, strat), start=1):
            ckpt_path = cv_dir / f"fold{fold}" / "best_model.pt"
            print(f"  Fold {fold}: {ckpt_path}  (val={len(val_idx)})")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            ck = ckpt["args"]
            tokenizer = AutoTokenizer.from_pretrained(ck["model_name"])
            model = MultiTaskClassifier(ck["model_name"], num_labels, float(ck.get("dropout", 0.1))).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            val_rows = [data[i] for i in val_idx]
            loader = DataLoader(TestDataset(val_rows, tokenizer, int(ck["max_len"])), batch_size=args.batch_size)
            local = 0
            with torch.no_grad():
                for batch in loader:
                    logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    bs = batch["input_ids"].size(0)
                    for f in EVAL_FIELDS:
                        probs = torch.softmax(logits[f], dim=-1).cpu()
                        for j in range(bs):
                            oof_prob[f][val_idx[local + j]] += probs[j]
                    local += bs
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Average over backbones, convert to log-prob for bias tuning (matches predict_test).
    n_backbones = len(args.cv_dir)
    oof_logprob = {f: (oof_prob[f] / n_backbones).clamp_min(1e-12).log() for f in EVAL_FIELDS}

    # Coordinate-ascent the per-class bias for each field.
    candidates = [round(b * 0.1, 1) for b in range(-30, 31)]  # -3.0 .. +3.0
    biases: dict[str, list[float]] = {}
    summary: dict[str, Any] = {"fields": {}}
    weighted_before = weighted_after = 0.0

    for field, labels in EVAL_FIELDS.items():
        y_true = [norm_label(x.get(field, ""), labels) for x in data]
        lp = oof_logprob[field]
        weight = {"promise_status": 0.20, "verification_timeline": 0.15,
                  "evidence_status": 0.30, "evidence_quality": 0.35}[field]

        def predict(bias: list[float]) -> list[str]:
            b = torch.tensor(bias)
            return [labels[int((lp[i] + b).argmax())] for i in range(n)]

        bias = [0.0] * len(labels)
        base_f1 = macro_f1(y_true, predict(bias), labels)
        best_f1 = base_f1
        for _ in range(4):  # passes of coordinate ascent
            for c in range(len(labels)):
                best_c, best_val = bias[c], best_f1
                for cand in candidates:
                    trial = list(bias)
                    trial[c] = cand
                    score = macro_f1(y_true, predict(trial), labels)
                    if score > best_val:
                        best_val, best_c = score, cand
                bias[c] = best_c
                best_f1 = best_val
        biases[field] = bias
        weighted_before += base_f1 * weight
        weighted_after += best_f1 * weight
        summary["fields"][field] = {
            "macro_f1_before": base_f1,
            "macro_f1_after": best_f1,
            "bias": {labels[i]: bias[i] for i in range(len(labels))},
        }
        print(f"  {field}: {base_f1:.4f} -> {best_f1:.4f}  bias={dict(zip(labels, bias))}")

    summary["weighted_before"] = weighted_before
    summary["weighted_after"] = weighted_after
    out = args.output or (args.cv_dir[0] / "class_bias.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump({"bias": biases, "summary": summary}, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print(f"OOF weighted_score: {weighted_before:.5f} -> {weighted_after:.5f} "
          f"(+{weighted_after - weighted_before:.5f})")
    print(f"Wrote biases -> {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
