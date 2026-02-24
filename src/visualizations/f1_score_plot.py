"""F1 Score visualization module.

This module provides plotting functions to visualize classification metrics
including precision, recall, and F1 score across different classification thresholds.
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_binary_classification_data(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load binary classification data from CSV file.

    Args:
        csv_path: Path to CSV file with 'predicted_probability' and 'actual' columns

    Returns:
        Tuple of (predicted_probabilities, actual_labels) as numpy arrays
    """
    predicted_probs = []
    actual_labels = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predicted_probs.append(float(row["predicted_probability"]))
            actual_labels.append(int(row["actual"]))

    return np.array(predicted_probs), np.array(actual_labels)


def plot_confusion_matrix(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives
        title: Plot title
        save_path: Optional path to save the figure
    """
    confusion = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(8, 6))
    im = plt.imshow(confusion, cmap="Blues", aspect="auto")

    # Set ticks and labels
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.title(title)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = plt.text(
                j,
                i,
                confusion[i, j],
                ha="center",
                va="center",
                color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.colorbar(im)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()


def plot_metrics_by_threshold(
    thresholds: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1_scores: np.ndarray,
    title: str = "Classification Metrics vs Threshold",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot precision, recall, and F1 score across different classification thresholds.

    Args:
        thresholds: Array of threshold values
        precisions: Precision values for each threshold
        recalls: Recall values for each threshold
        f1_scores: F1 score values for each threshold
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        precisions,
        marker="o",
        label="Precision",
        linewidth=2,
        markersize=4,
    )
    plt.plot(
        thresholds,
        recalls,
        marker="s",
        label="Recall",
        linewidth=2,
        markersize=4,
    )
    plt.plot(
        thresholds,
        f1_scores,
        marker="^",
        label="F1 Score",
        linewidth=2,
        markersize=4,
        color="green",
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(8, 6))

    # Plot ROC curve
    plt.plot(fpr, tpr, color="blue", linewidth=2, label="ROC Curve")

    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
