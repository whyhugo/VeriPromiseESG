#!/usr/bin/env python3
"""Train the VeriPromiseESG multi-task text classification baseline.

The script is a runnable version of the official notebook baseline with:
- direct loading from data/
- label normalization for released files that use "" instead of "N/A"
- structured experiment artifacts under experiments/<run_name>/
- optional TensorBoard scalar logging
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


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

FIELD_WEIGHTS: dict[str, float] = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-task Transformer baseline for VeriPromiseESG."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/vpesg_4k_train_1000.json"),
        help="Training JSON path. Defaults to the data/ copy.",
    )
    parser.add_argument(
        "--merge-val",
        action="store_true",
        help="Append the official validation file to the labeled training pool "
        "(1000+1000=2000 labeled samples).",
    )
    parser.add_argument(
        "--val-data-path",
        type=Path,
        default=Path("data/vpesg4k_val_1000.json"),
        help="Official validation JSON, merged into training when --merge-val is set.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >1, run stratified K-fold cross-validation instead of a single "
        "hold-out split, and report mean/std across folds.",
    )
    parser.add_argument(
        "--cv-stratify-field",
        default="evidence_status",
        choices=list(EVAL_FIELDS.keys()),
        help="Field used to stratify CV folds (default evidence_status; "
        "avoids evidence_quality whose Misleading class has too few samples).",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Train on ALL labeled data with no validation split (final "
        "submission model). Trains for the full --epochs and saves the last epoch.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--experiment-name", default="bert_baseline")
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument(
        "--limit-data",
        type=int,
        default=0,
        help="Use only the first N samples. Intended for smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Stop each epoch after N training batches. Intended for smoke tests.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--stratify-field",
        default="",
        choices=["", *EVAL_FIELDS.keys()],
        help="Optional field for train/val stratification. Empty matches notebook split.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Use balanced class weights per task to reduce majority-label bias.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard logs to <run_dir>/tensorboard.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA automatic mixed precision when CUDA is available.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG plots. Metrics JSON/CSV are still written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data, normalize labels, create a run folder, then exit before ML imports.",
    )
    return parser.parse_args()


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(sample)
    for field, labels in EVAL_FIELDS.items():
        value = normalized.get(field, "")
        if value in ("", None) and "N/A" in labels:
            value = "N/A"
        if value not in labels:
            raise ValueError(
                f"Unknown label for {field}: {value!r}. "
                f"Expected one of {labels}. Sample id={sample.get('id')}"
            )
        normalized[field] = value
    return normalized


def load_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [normalize_sample(item) for item in data]


def build_label_maps() -> tuple[dict[str, dict[str, int]], dict[str, dict[int, str]]]:
    label2id = {
        field: {label: idx for idx, label in enumerate(labels)}
        for field, labels in EVAL_FIELDS.items()
    }
    id2label = {
        field: {idx: label for idx, label in enumerate(labels)}
        for field, labels in EVAL_FIELDS.items()
    }
    return label2id, id2label


def make_run_dir(output_dir: Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def label_distribution(data: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        field: {label: Counter(item[field] for item in data).get(label, 0) for label in labels}
        for field, labels in EVAL_FIELDS.items()
    }


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    rows: list[dict[str, Any]] = []
    for item in history:
        row = {
            "epoch": item["epoch"],
            "train_loss": item["train_loss"],
            "weighted_score": item["weighted_score"],
            "lr": item["lr"],
        }
        for field, metrics in item["fields"].items():
            row[f"{field}_macro_f1"] = metrics["macro_f1"]
            row[f"{field}_micro_f1"] = metrics["micro_f1"]
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_config(args: argparse.Namespace, run_dir: Path, data: list[dict[str, Any]]) -> None:
    config = vars(args).copy()
    config["data_path"] = str(config["data_path"])
    config["output_dir"] = str(config["output_dir"])
    config["val_data_path"] = str(config["val_data_path"])
    config["run_dir"] = str(run_dir)
    config["eval_fields"] = EVAL_FIELDS
    config["field_weights"] = FIELD_WEIGHTS
    config["data_size"] = len(data)
    config["label_distribution"] = label_distribution(data)
    write_json(run_dir / "config.json", config)


def run_training(
    args: argparse.Namespace,
    run_dir: Path,
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    all_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train on train_data, evaluate on val_data, return final metrics.

    When val_data is empty (full-train mode) the last epoch is saved and no
    evaluation is performed.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import classification_report, f1_score
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

    if all_data is None:
        all_data = train_data + val_data
    full_train = len(val_data) == 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    label2id, id2label = build_label_maps()
    num_labels = {field: len(labels) for field, labels in EVAL_FIELDS.items()}

    write_json(run_dir / "split_summary.json", {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "train_distribution": label_distribution(train_data),
        "val_distribution": label_distribution(val_data) if val_data else {},
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class ESGDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            sample = self.rows[idx]
            encoding = tokenizer(
                sample["data"],
                truncation=True,
                max_length=args.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            labels = {
                field: torch.tensor(label2id[field][sample[field]], dtype=torch.long)
                for field in EVAL_FIELDS
            }
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": labels,
            }

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": {
                field: torch.stack([item["labels"][field] for item in batch])
                for field in EVAL_FIELDS
            },
        }

    train_loader = DataLoader(
        ESGDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = None
    if not full_train:
        val_loader = DataLoader(
            ESGDataset(val_data),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    class MultiTaskClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(args.model_name)
            # Some checkpoints (e.g. Erlangshen DeBERTa-v2) ship fp16 weights, which
            # clashes with the fp32 heads. Force fp32; no-op for fp32 backbones.
            self.encoder = self.encoder.float()
            hidden_size = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(args.dropout)
            self.classifiers = nn.ModuleDict({
                field: nn.Linear(hidden_size, count)
                for field, count in num_labels.items()
            })

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            return {field: head(pooled) for field, head in self.classifiers.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_steps_per_epoch = len(train_loader)
    if args.max_train_batches > 0:
        train_steps_per_epoch = min(train_steps_per_epoch, args.max_train_batches)
    total_steps = train_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    class_weights: dict[str, torch.Tensor | None] = {}
    for field, labels in EVAL_FIELDS.items():
        if args.class_weight == "balanced":
            counts = Counter(item[field] for item in train_data)
            total = sum(counts.values())
            weights = [
                total / (len(labels) * max(counts.get(label, 0), 1))
                for label in labels
            ]
            class_weights[field] = torch.tensor(weights, dtype=torch.float, device=device)
        else:
            class_weights[field] = None

    criteria = {
        field: nn.CrossEntropyLoss(weight=class_weights[field])
        for field in EVAL_FIELDS
    }

    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print("TensorBoard is not available. Install tensorboard to enable logging.")
        else:
            writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def evaluate(predictions: list[dict[str, str]], gt_rows: list[dict[str, Any]]) -> dict[str, Any]:
        results: dict[str, Any] = {"fields": {}, "weighted_score": 0.0}
        for field, labels in EVAL_FIELDS.items():
            y_true = [item[field] for item in gt_rows]
            y_pred = [item[field] for item in predictions]
            macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
            micro = float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))
            report_dict = classification_report(
                y_true,
                y_pred,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            report_text = classification_report(
                y_true,
                y_pred,
                labels=labels,
                zero_division=0,
            )
            weight = FIELD_WEIGHTS[field]
            results["weighted_score"] = float(results["weighted_score"] + macro * weight)
            results["fields"][field] = {
                "macro_f1": macro,
                "micro_f1": micro,
                "weight": weight,
                "report": report_dict,
                "report_text": report_text,
            }
        return results

    def predict(loader: DataLoader) -> list[dict[str, str]]:
        model.eval()
        predictions: list[dict[str, str]] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_size = input_ids.size(0)
                for idx in range(batch_size):
                    item: dict[str, str] = {}
                    for field in EVAL_FIELDS:
                        pred_id = int(logits[field][idx].argmax().item())
                        item[field] = id2label[field][pred_id]
                    predictions.append(item)
        return predictions

    def train_one_epoch(epoch: int) -> float:
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = {
                field: batch["labels"][field].to(device)
                for field in EVAL_FIELDS
            }

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = sum(criteria[field](logits[field], labels[field]) for field in EVAL_FIELDS)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.item())
            global_step = (epoch - 1) * len(train_loader) + step
            if writer:
                writer.add_scalar("train/loss_step", float(loss.item()), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            if step % 50 == 0 or step == train_steps_per_epoch:
                print(f"Epoch {epoch} step {step}/{train_steps_per_epoch} loss={loss.item():.4f}")
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break
        return total_loss / max(train_steps_per_epoch, 1)

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_model_path = run_dir / "best_model.pt"

    def save_checkpoint(epoch: int, score: float) -> None:
        checkpoint_args = vars(args).copy()
        for key in ("data_path", "output_dir", "val_data_path"):
            if key in checkpoint_args:
                checkpoint_args[key] = str(checkpoint_args[key])
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": checkpoint_args,
                "label2id": label2id,
                "id2label": id2label,
                "best_score": float(score),
                "best_epoch": epoch,
            },
            best_model_path,
        )

    if full_train:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(epoch)
            print(f"Epoch {epoch}: loss={train_loss:.4f} (full-train, no val)")
            history.append({"epoch": epoch, "train_loss": train_loss})
            if writer:
                writer.add_scalar("train/loss_epoch", train_loss, epoch)
            write_json(run_dir / "history.json", history)
        save_checkpoint(args.epochs, -1.0)
        final_metrics = {
            "weighted_score": None,
            "best_epoch": args.epochs,
            "best_model_path": str(best_model_path),
            "note": "full-train mode: trained on all labeled data, no validation.",
        }
        write_json(run_dir / "metrics.json", final_metrics)
        if writer:
            writer.close()
        print("=" * 72)
        print(f"Run directory: {run_dir} (full-train, {len(train_data)} samples)")
        print("=" * 72)
        return final_metrics

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(epoch)
        val_predictions = predict(val_loader)
        metrics = evaluate(val_predictions, val_data)
        weighted = metrics["weighted_score"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "weighted_score": weighted,
            "lr": scheduler.get_last_lr()[0],
            "fields": {
                field: {
                    "macro_f1": metrics["fields"][field]["macro_f1"],
                    "micro_f1": metrics["fields"][field]["micro_f1"],
                }
                for field in EVAL_FIELDS
            },
        }
        history.append(row)

        print(f"Epoch {epoch}: loss={train_loss:.4f} weighted_score={weighted:.5f}")
        for field in EVAL_FIELDS:
            field_metrics = metrics["fields"][field]
            print(
                f"  {field}: macro_f1={field_metrics['macro_f1']:.4f} "
                f"micro_f1={field_metrics['micro_f1']:.4f}"
            )

        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("val/weighted_score", weighted, epoch)
            for field in EVAL_FIELDS:
                writer.add_scalar(
                    f"val/{field}_macro_f1",
                    metrics["fields"][field]["macro_f1"],
                    epoch,
                )
                writer.add_scalar(
                    f"val/{field}_micro_f1",
                    metrics["fields"][field]["micro_f1"],
                    epoch,
                )

        if weighted > best_score:
            best_score = weighted
            best_epoch = epoch
            save_checkpoint(best_epoch, best_score)

        write_json(run_dir / "history.json", history)
        write_history_csv(run_dir / "history.csv", history)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_predictions = predict(val_loader)
    final_metrics = evaluate(final_predictions, val_data)
    final_metrics["best_epoch"] = best_epoch
    final_metrics["best_model_path"] = str(best_model_path)

    write_json(run_dir / "metrics.json", final_metrics)
    write_json(run_dir / "predictions_val.json", [
        {
            "id": row.get("id"),
            "ground_truth": {field: row[field] for field in EVAL_FIELDS},
            "prediction": pred,
            "data": row.get("data", ""),
        }
        for row, pred in zip(val_data, final_predictions)
    ])
    write_json(run_dir / "prediction_submission_like.json", [
        {"id": row.get("id"), **pred}
        for row, pred in zip(val_data, final_predictions)
    ])

    if not args.skip_plots:
        save_plots(run_dir, all_data, train_data, history, final_metrics)

    if writer:
        writer.close()

    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Best epoch: {best_epoch}")
    print(f"Final weighted score: {final_metrics['weighted_score']:.5f}")
    print("=" * 72)
    return final_metrics


def save_plots(
    run_dir: Path,
    all_data: list[dict[str, Any]],
    train_data: list[dict[str, Any]],
    history: list[dict[str, Any]],
    final_metrics: dict[str, Any],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        print(f"Skipping plots because matplotlib/seaborn is unavailable: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training data label distribution", fontsize=16, fontweight="bold")
    distribution = label_distribution(train_data)
    for idx, (field, labels) in enumerate(EVAL_FIELDS.items()):
        ax = axes[idx // 2][idx % 2]
        counts = distribution[field]
        bars = ax.bar(labels, [counts[label] for label in labels], color=sns.color_palette("husl", len(labels)))
        ax.set_title(field)
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)
        total = max(sum(counts.values()), 1)
        for bar, label in zip(bars, labels):
            count = counts[label]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count}\n({count / total * 100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(run_dir / "label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = [item["epoch"] for item in history]
    axes[0].plot(epochs, [item["train_loss"] for item in history], "b-o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [item["weighted_score"] for item in history], "g-o")
    axes[1].set_title("Validation Weighted Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted F1")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fields = list(EVAL_FIELDS)
    macro_f1s = [final_metrics["fields"][field]["macro_f1"] for field in fields]
    micro_f1s = [final_metrics["fields"][field]["micro_f1"] for field in fields]
    x = list(range(len(fields)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width / 2 for i in x], macro_f1s, width, label="Macro F1", color="steelblue")
    ax.bar([i + width / 2 for i in x], micro_f1s, width, label="Micro F1", color="coral")
    ax.axhline(final_metrics["weighted_score"], color="green", linestyle="--", label="Weighted score")
    ax.set_title(f"F1 Score by Task (weighted={final_metrics['weighted_score']:.5f})")
    ax.set_xlabel("Task")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(run_dir / "f1_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lengths = [len(item["data"]) for item in all_data]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50, color="steelblue", alpha=0.8, edgecolor="black")
    ax.axvline(256, color="red", linestyle="--", label="MAX_LEN=256")
    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "text_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_cross_validation(
    args: argparse.Namespace, run_dir: Path, data: list[dict[str, Any]]
) -> None:
    from sklearn.model_selection import StratifiedKFold

    strat = [item[args.cv_stratify_field] for item in data]
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    fold_metrics: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, strat), start=1):
        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        print(f"\n{'#' * 72}\n# Fold {fold}/{args.cv_folds}  "
              f"train={len(train_data)} val={len(val_data)}\n{'#' * 72}")
        metrics = run_training(args, fold_dir, train_data, val_data, all_data=data)
        fold_metrics.append(metrics)

    import statistics
    scores = [m["weighted_score"] for m in fold_metrics]
    per_field = {
        field: [m["fields"][field]["macro_f1"] for m in fold_metrics]
        for field in EVAL_FIELDS
    }
    summary = {
        "cv_folds": args.cv_folds,
        "weighted_score_mean": statistics.mean(scores),
        "weighted_score_std": statistics.pstdev(scores),
        "weighted_score_per_fold": scores,
        "per_field": {
            field: {
                "macro_f1_mean": statistics.mean(vals),
                "macro_f1_std": statistics.pstdev(vals),
                "per_fold": vals,
            }
            for field, vals in per_field.items()
        },
    }
    write_json(run_dir / "cv_summary.json", summary)
    print("\n" + "=" * 72)
    print(f"CV {args.cv_folds}-fold weighted_score: "
          f"{summary['weighted_score_mean']:.5f} ± {summary['weighted_score_std']:.5f}")
    for field, stats in summary["per_field"].items():
        print(f"  {field}: {stats['macro_f1_mean']:.4f} ± {stats['macro_f1_std']:.4f}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    data = load_data(args.data_path)
    if args.merge_val:
        val_extra = load_data(args.val_data_path)
        data = data + val_extra
        print(f"Merged validation set: +{len(val_extra)} -> {len(data)} labeled samples.")
    if args.limit_data > 0:
        data = data[:args.limit_data]
    run_dir = make_run_dir(args.output_dir, args.experiment_name)
    save_config(args, run_dir, data)
    write_json(run_dir / "data_summary.json", {
        "size": len(data),
        "label_distribution": label_distribution(data),
        "text_length": {
            "min": min(len(item["data"]) for item in data),
            "max": max(len(item["data"]) for item in data),
            "avg": sum(len(item["data"]) for item in data) / max(len(data), 1),
            "over_256": sum(len(item["data"]) > 256 for item in data),
        },
    })
    if args.dry_run:
        print(f"Dry run complete. Run directory: {run_dir}")
        return

    if args.cv_folds and args.cv_folds > 1:
        run_cross_validation(args, run_dir, data)
        return

    if args.full_train:
        run_training(args, run_dir, data, [], all_data=data)
        return

    from sklearn.model_selection import train_test_split
    stratify = [item[args.stratify_field] for item in data] if args.stratify_field else None
    train_data, val_data = train_test_split(
        data, test_size=args.val_size, random_state=args.seed, stratify=stratify,
    )
    run_training(args, run_dir, train_data, val_data, all_data=data)


if __name__ == "__main__":
    main()
