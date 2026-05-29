"""Visualization utilities for PersuasiX analysis and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    save_path: str | None = None,
    figsize: tuple = (14, 12),
) -> plt.Figure:
    """Plot multi-label confusion matrix (per-technique)."""
    from sklearn.metrics import multilabel_confusion_matrix

    mcm = multilabel_confusion_matrix(y_true, y_pred)

    n_labels = len(labels)
    cols = 4
    rows = (n_labels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, (cm, label) in enumerate(zip(mcm, labels)):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
        )
        axes[i].set_title(label.replace("_", " ").title(), fontsize=9)
        axes[i].set_ylabel("True")
        axes[i].set_xlabel("Pred")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Technique Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_technique_distribution(
    technique_counts: dict[str, int],
    save_path: str | None = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Bar chart of technique frequency distribution."""
    sorted_items = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", len(labels))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Count")
    ax.set_title("Persuasion Technique Distribution", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | None = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot loss curves and learning rate schedule."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Validation", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "learning_rate" in history:
        axes[1].plot(epochs, history["learning_rate"], "g-o", markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].grid(True, alpha=0.3)

    if "epoch_time" in history:
        axes[2].bar(epochs, history["epoch_time"], color="steelblue", alpha=0.7)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Time (s)")
        axes[2].set_title("Epoch Duration")
        axes[2].grid(True, alpha=0.3)

    fig.suptitle("PersuasiX Training Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_language_breakdown(
    language_counts: dict[str, int],
    save_path: str | None = None,
) -> plt.Figure:
    """Pie chart for language distribution."""
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = [{"en": "English", "fr": "French", "ar": "Arabic"}.get(k, k) for k in language_counts]
    values = list(language_counts.values())
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)],
           startangle=90, textprops={"fontsize": 12})
    ax.set_title("Dataset Language Distribution", fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_severity_distribution(
    severity_scores: list[float],
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of manipulation severity scores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(severity_scores, bins=20, color="coral", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Severity Score")
    ax.set_ylabel("Count")
    ax.set_title("Manipulation Severity Distribution", fontsize=14, fontweight="bold")
    ax.axvline(np.mean(severity_scores), color="red", linestyle="--",
               label=f"Mean: {np.mean(severity_scores):.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
