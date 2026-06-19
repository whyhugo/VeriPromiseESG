#!/usr/bin/env python3
"""Generate an AIdea submission CSV from trained VeriPromiseESG checkpoints.

Loads one or more best_model.pt checkpoints, predicts the (unlabeled) test set,
and writes a CSV matching data/sample_submission_format.csv:

    id,promise_status,verification_timeline,evidence_status,evidence_quality

With multiple checkpoints the per-field softmax probabilities are averaged
(soft-voting ensemble) before argmax.

Examples
--------
# single model
python3 scripts/predict_test.py \
    --checkpoint experiments/<run>/best_model.pt \
    --output submissions/submission.csv

# 5-seed / 5-fold ensemble
python3 scripts/predict_test.py \
    --checkpoint experiments/<runA>/best_model.pt experiments/<runB>/.../best_model.pt \
    --output submissions/ensemble.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


EVAL_FIELDS: dict[str, list[str]] = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}
SUBMISSION_COLUMNS = ["id", *EVAL_FIELDS.keys()]


class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: dict[str, int], dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Some checkpoints (e.g. Erlangshen DeBERTa-v2) ship fp16 weights; force fp32.
        self.encoder = self.encoder.float()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleDict(
            {field: nn.Linear(hidden_size, count) for field, count in num_labels.items()}
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {field: head(pooled) for field, head in self.classifiers.items()}


class TestDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_len: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.rows[idx]["data"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict the test set and write an AIdea submission CSV.")
    p.add_argument("--checkpoint", nargs="+", required=True, help="One or more best_model.pt paths.")
    p.add_argument("--test-path", type=Path, default=Path("data/vpesg4k_test_2000.json"))
    p.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--bias-file",
        type=Path,
        default=None,
        help="Optional class_bias.json from tune_thresholds.py. Per-class biases "
        "are added to the averaged log-prob before argmax.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with args.test_path.open("r", encoding="utf-8") as f:
        test_rows = json.load(f)
    ids = [row.get("id") for row in test_rows]
    num_samples = len(test_rows)

    num_labels = {field: len(labels) for field, labels in EVAL_FIELDS.items()}
    # Accumulated softmax probabilities per field, averaged over checkpoints.
    prob_sum = {field: torch.zeros(num_samples, num_labels[field]) for field in EVAL_FIELDS}

    for ckpt_path in args.checkpoint:
        print(f"Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ck_args = ckpt["args"]
        model_name = ck_args["model_name"]
        max_len = int(ck_args["max_len"])
        dropout = float(ck_args.get("dropout", 0.1))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MultiTaskClassifier(model_name, num_labels, dropout).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        loader = DataLoader(
            TestDataset(test_rows, tokenizer, max_len),
            batch_size=args.batch_size,
            shuffle=False,
        )
        offset = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                bs = input_ids.size(0)
                for field in EVAL_FIELDS:
                    probs = torch.softmax(logits[field], dim=-1).cpu()
                    prob_sum[field][offset:offset + bs] += probs
                offset += bs
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bias = {field: torch.zeros(num_labels[field]) for field in EVAL_FIELDS}
    if args.bias_file:
        with args.bias_file.open(encoding="utf-8") as f:
            loaded = json.load(f)["bias"]
        for field in EVAL_FIELDS:
            bias[field] = torch.tensor(loaded[field], dtype=torch.float)
        print(f"Applied per-class biases from {args.bias_file}")

    # Average over checkpoints, then bias in log space before argmax.
    avg_logprob = {field: (prob_sum[field] / len(args.checkpoint)).clamp_min(1e-12).log()
                   for field in EVAL_FIELDS}
    predictions: list[dict[str, str]] = []
    for i in range(num_samples):
        row = {"id": ids[i]}
        for field, labels in EVAL_FIELDS.items():
            row[field] = labels[int((avg_logprob[field][i] + bias[field]).argmax().item())]
        predictions.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Wrote {len(predictions)} rows from {len(args.checkpoint)} checkpoint(s) -> {args.output}")


if __name__ == "__main__":
    main()
