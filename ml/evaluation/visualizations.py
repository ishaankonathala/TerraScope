"""Evaluation charts for loss curves, confusion matrices, and model comparisons."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Training & Validation Curves",
) -> Path | None:
    """Plot loss and F1/IoU curves from logged epoch history."""
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", 0.0) for h in history]
    val_loss = [h.get("val_loss", 0.0) for h in history]
    val_f1 = [h.get("val_f1", 0.0) for h in history]
    val_iou = [h.get("val_iou", 0.0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    axes[0].plot(epochs, train_loss, label="Train Loss", marker="o", linewidth=1.8)
    axes[0].plot(epochs, val_loss, label="Val Loss", marker="s", linewidth=1.8)
    axes[0].set_title("Loss Progression")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="Val F1", marker="o", linewidth=1.8)
    axes[1].plot(epochs, val_iou, label="Val IoU", marker="^", linewidth=1.8)
    axes[1].set_title("Validation F1 & IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    output_path: str | Path,
    title: str = "Pixel-Level Confusion Matrix",
) -> Path:
    """Save a readable 2x2 confusion matrix figure."""
    arr = np.asarray(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=140)
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Unchanged", "Changed"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    thresh = arr.max() / 2.0 if arr.size else 0
    for i in range(2):
        for j in range(2):
            color = "white" if arr[i, j] > thresh else "black"
            ax.text(j, i, f"{arr[i, j]:,}", ha="center", va="center", color=color, fontsize=10)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_model_comparison(
    benchmark_payload: dict[str, Any],
    output_path: str | Path,
) -> Path | None:
    """Save grouped bar comparison chart from actual benchmark evaluation output."""
    models = benchmark_payload.get("models", [])
    if not models:
        return None

    names = [m["display_name"] for m in models]
    metric_keys = ["precision", "recall", "f1_score", "iou", "accuracy"]
    labels = ["Precision", "Recall", "F1", "IoU", "Accuracy"]

    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    for idx, (k, lbl) in enumerate(zip(metric_keys, labels)):
        vals = [float(m["metrics"].get(k, 0.0)) for m in models]
        offset = (idx - len(metric_keys) / 2.0 + 0.5) * width
        ax.bar(x + offset, vals, width, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Architecture Comparison — {benchmark_payload.get('dataset_name', '')} "
        f"({benchmark_payload.get('split', '')})"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

