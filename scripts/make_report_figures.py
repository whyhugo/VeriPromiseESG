#!/usr/bin/env python3
"""Generate report-quality figures from retained CV training artifacts.

Outputs PNGs to reports/figures/. Axis labels are kept in English to avoid
CJK font issues across environments.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)

BACKBONES = {
    "RoBERTa-large": "experiments/*roberta_large_mergeval_cv5",
    "MacBERT-large": "experiments/*macbert_large_mergeval_cv5",
    "DeBERTa-v2-320M": "experiments/*deberta_v2_320m_mergeval_cv5",
}
FIELDS = ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]

EVAL_FIELDS = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}


def dir_of(pattern: str) -> str:
    return sorted(glob.glob(pattern))[-1]


def load_cv(pattern: str) -> dict:
    return json.load(open(os.path.join(dir_of(pattern), "cv_summary.json")))


# ---- Fig 1: training curves (RoBERTa, mean ± std over 5 folds) ----
def fig_training_curve():
    rob = dir_of(BACKBONES["RoBERTa-large"])
    losses, scores = [], []
    for f in sorted(glob.glob(os.path.join(rob, "fold*/history.json"))):
        h = json.load(open(f))
        losses.append([e["train_loss"] for e in h])
        scores.append([e["weighted_score"] for e in h])
    losses, scores = np.array(losses), np.array(scores)
    epochs = np.arange(1, losses.shape[1] + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for arr, a, title, color in [
        (losses, ax[0], "Training Loss", "tab:red"),
        (scores, ax[1], "Validation Weighted Score", "tab:green"),
    ]:
        m, s = arr.mean(0), arr.std(0)
        a.plot(epochs, m, "-o", color=color, label="mean over 5 folds")
        a.fill_between(epochs, m - s, m + s, alpha=0.2, color=color, label="± std")
        a.set_xlabel("Epoch"); a.set_title(title); a.grid(alpha=0.3)
    # mark the mean best-epoch (where val weighted score peaks per fold)
    best_ep = float(np.mean([int(np.argmax(s)) + 1 for s in scores]))
    ax[1].axhline(0.6224, ls="--", c="gray", lw=1, label="CV mean (best-epoch)")
    ax[1].axvline(best_ep, ls=":", c="black", lw=1.5, label=f"mean best epoch ≈ {best_ep:.0f}")
    ax[0].legend(); ax[1].legend()
    fig.suptitle("RoBERTa-large 5-fold CV training dynamics", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_training_curve.png", dpi=150); plt.close(fig)


# ---- Fig 2: CV weighted score per backbone (mean ± std) ----
def fig_backbone_cv():
    names, means, stds = [], [], []
    for name, pat in BACKBONES.items():
        cv = load_cv(pat)
        names.append(name); means.append(cv["weighted_score_mean"]); stds.append(cv["weighted_score_std"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=["tab:blue", "tab:orange", "tab:green"], alpha=0.85)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.004, f"{m:.4f}", ha="center", fontsize=10)
    ax.set_ylabel("5-fold CV weighted score (OOF)")
    ax.set_ylim(0.55, 0.65); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Single-backbone CV performance (mean ± std)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_backbone_cv.png", dpi=150); plt.close(fig)


# ---- Fig 3: per-field Macro-F1 by backbone ----
def fig_perfield():
    data = {name: load_cv(pat)["per_field"] for name, pat in BACKBONES.items()}
    x = np.arange(len(FIELDS)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (name, color) in enumerate(zip(BACKBONES, ["tab:blue", "tab:orange", "tab:green"])):
        vals = [data[name][f]["macro_f1_mean"] for f in FIELDS]
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(FIELDS, rotation=12)
    ax.set_ylabel("Macro-F1 (CV mean)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-task Macro-F1 by backbone — evidence_quality is the bottleneck", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_perfield_by_backbone.png", dpi=150); plt.close(fig)


# ---- Fig 4: evidence_quality per-class F1 (RoBERTa, mean over folds) ----
def fig_evq_perclass():
    rob = dir_of(BACKBONES["RoBERTa-large"])
    classes = ["Clear", "Not Clear", "Misleading", "N/A"]
    per = {c: [] for c in classes}
    for f in sorted(glob.glob(os.path.join(rob, "fold*/metrics.json"))):
        rep = json.load(open(f))["fields"]["evidence_quality"]["report"]
        for c in classes:
            per[c].append(rep.get(c, {}).get("f1-score", 0.0))
    means = [np.mean(per[c]) for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:green", "tab:olive", "tab:red", "tab:gray"]
    bars = ax.bar(classes, means, color=colors, alpha=0.85)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.01, f"{m:.2f}", ha="center")
    ax.set_ylabel("F1-score (CV mean)"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    ax.set_title("evidence_quality per-class F1 — 'Misleading' (2 samples) stuck at 0", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_evidence_quality_perclass.png", dpi=150); plt.close(fig)


# ---- Fig 5: OOF vs Leaderboard calibration ----
def fig_calibration():
    labels = ["RoBERTa\n(raw)", "rob+mac\n(tuned)", "3-way\n(tuned)", "3-way\n(raw, final)"]
    oof = [0.6239, 0.6366, 0.6414, 0.6323]
    lb = [0.6168, 0.6117, 0.6172, 0.6176]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, oof, w, label="Local OOF", color="tab:blue", alpha=0.85)
    ax.bar(x + w / 2, lb, w, label="Public LB", color="tab:red", alpha=0.85)
    for i in range(len(labels)):
        ax.text(x[i] - w / 2, oof[i] + 0.001, f"{oof[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, lb[i] + 0.001, f"{lb[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.60, 0.65); ax.set_ylabel("weighted score"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Calibration: threshold tuning inflates OOF but not LB", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_calibration.png", dpi=150); plt.close(fig)


# ---- Fig 6: confusion matrices (RoBERTa OOF, aggregated over 5 folds) ----
def fig_confusion():
    from sklearn.metrics import confusion_matrix
    rob = dir_of(BACKBONES["RoBERTa-large"])
    gts = {f: [] for f in FIELDS}
    prs = {f: [] for f in FIELDS}
    for pf in sorted(glob.glob(os.path.join(rob, "fold*/predictions_val.json"))):
        for row in json.load(open(pf)):
            for f in FIELDS:
                gts[f].append(row["ground_truth"][f])
                prs[f].append(row["prediction"][f])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, f in zip(axes.ravel(), FIELDS):
        labels = EVAL_FIELDS[f]
        cm = confusion_matrix(gts[f], prs[f], labels=labels)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right"); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f, fontweight="bold")
        thresh = cm.max() / 2 if cm.max() else 1
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=11)
    fig.suptitle("Confusion matrices — RoBERTa OOF (2,000 samples, 5-fold aggregated)",
                 fontweight="bold", fontsize=14)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_confusion_matrices.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_training_curve()
    fig_backbone_cv()
    fig_perfield()
    fig_evq_perclass()
    fig_calibration()
    fig_confusion()
    print("Wrote figures to", OUT)
    for p in sorted(glob.glob(f"{OUT}/*.png")):
        print(" ", p)
