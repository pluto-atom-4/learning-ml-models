"""F1 Score visualization runner.

This module demonstrates F1 score calculation and visualization by:
1. Creating a binary classification dataset with predicted probabilities
2. Computing precision, recall, and F1 score across different thresholds
3. Generating visualizations of the metrics
"""

from pathlib import Path
import sys

import numpy as np

# Add parent directories to path to enable src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.machine_learning.f1_score import precision, recall, f1_score
from src.visualizations.f1_score_plot import (
    plot_confusion_matrix,
    plot_metrics_by_threshold,
    plot_roc_curve,
)


def generate_synthetic_predictions(
    n_samples: int = 200, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic predicted probabilities and actual labels.

    Args:
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (predicted_probabilities, actual_labels)
    """
    np.random.seed(random_seed)

    # Generate predicted probabilities
    y_pred_probs = np.random.uniform(0, 1, n_samples)

    # Generate actual labels with some correlation to predicted probabilities
    y_actual = (y_pred_probs + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)

    return y_pred_probs, y_actual


def compute_metrics_across_thresholds(
    y_pred_probs: np.ndarray,
    y_actual: np.ndarray,
    num_thresholds: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision, recall, and F1 score across different classification thresholds.

    Args:
        y_pred_probs: Predicted probabilities
        y_actual: Actual labels
        num_thresholds: Number of thresholds to evaluate

    Returns:
        Tuple of (thresholds, precisions, recalls, f1_scores)
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)

        p = precision(y_actual, y_pred)
        r = recall(y_actual, y_pred)
        f1 = f1_score(y_actual, y_pred)

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    return thresholds, np.array(precisions), np.array(recalls), np.array(f1_scores)


def compute_confusion_matrix(
    y_actual: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Compute confusion matrix values.

    Args:
        y_actual: Actual labels
        y_pred: Predicted labels

    Returns:
        Tuple of (tp, fp, tn, fn)
    """
    tp = np.sum((y_actual == 1) & (y_pred == 1))
    fp = np.sum((y_actual == 0) & (y_pred == 1))
    tn = np.sum((y_actual == 0) & (y_pred == 0))
    fn = np.sum((y_actual == 1) & (y_pred == 0))

    return int(tp), int(fp), int(tn), int(fn)


def compute_roc_curve(
    y_actual: np.ndarray,
    y_pred_probs: np.ndarray,
    num_thresholds: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC curve (False Positive Rate vs True Positive Rate).

    Args:
        y_actual: Actual labels
        y_pred_probs: Predicted probabilities
        num_thresholds: Number of thresholds to evaluate

    Returns:
        Tuple of (false_positive_rates, true_positive_rates)
    """
    thresholds = np.linspace(1, 0, num_thresholds)
    tprs = []
    fprs = []

    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        tp, fp, tn, fn = compute_confusion_matrix(y_actual, y_pred)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    return np.array(fprs), np.array(tprs)


def main() -> None:
    """
    Generate synthetic data, compute metrics, and create visualizations.
    """
    print("=" * 60)
    print("F1 Score Visualization Demo")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic binary classification data...")
    y_pred_probs, y_actual = generate_synthetic_predictions(n_samples=200)
    print(f"   Generated {len(y_actual)} samples")
    print(f"   Positive class proportion: {y_actual.mean():.2%}")

    # Use threshold = 0.5 for main metrics
    threshold = 0.5
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Compute metrics at threshold 0.5
    print(f"\n2. Computing metrics at threshold = {threshold}...")
    p = precision(y_actual, y_pred)
    r = recall(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)

    print(f"   Precision: {p:.4f}")
    print(f"   Recall:    {r:.4f}")
    print(f"   F1 Score:  {f1:.4f}")

    # Compute confusion matrix
    tp, fp, tn, fn = compute_confusion_matrix(y_actual, y_pred)
    print(f"\n3. Confusion Matrix (at threshold = {threshold}):")
    print(f"   True Positives:  {tp}")
    print(f"   False Positives: {fp}")
    print(f"   True Negatives:  {tn}")
    print(f"   False Negatives: {fn}")

    # Compute metrics across thresholds
    print("\n4. Computing metrics across 11 different thresholds...")
    thresholds, precisions, recalls, f1_scores = compute_metrics_across_thresholds(
        y_pred_probs, y_actual, num_thresholds=11
    )
    print(f"   Threshold range: {thresholds.min():.1f} to {thresholds.max():.1f}")
    print(f"   Best F1 Score: {f1_scores.max():.4f}")
    best_threshold_idx = np.argmax(f1_scores)
    print(f"   Best threshold: {thresholds[best_threshold_idx]:.2f}")

    # Compute ROC curve
    print("\n5. Computing ROC curve...")
    fpr, tpr = compute_roc_curve(y_actual, y_pred_probs, num_thresholds=100)

    # Create visualizations
    print("\n6. Creating visualizations...")

    # Plot 1: Confusion Matrix
    plot_confusion_matrix(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        title=f"Confusion Matrix (threshold={threshold})",
    )

    # Plot 2: Metrics across thresholds
    plot_metrics_by_threshold(
        thresholds,
        precisions,
        recalls,
        f1_scores,
        title="Classification Metrics vs Threshold",
    )

    # Plot 3: ROC Curve
    plot_roc_curve(fpr, tpr, title="ROC Curve")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
