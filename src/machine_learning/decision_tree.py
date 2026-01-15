"""
Decision Tree Classifier for classification tasks.

Implements a classification tree using impurity-based splits (Gini or Entropy).
Inherits common tree functionality from BaseDecisionTree.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Callable

from .base_tree import BaseDecisionTree


# =====================================================================
# Impurity Metrics (Classification-specific)
# =====================================================================
def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity for a set of labels."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)


def entropy(y: np.ndarray) -> float:
    """Compute entropy (information content) for a set of labels."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def information_gain(y, y_left, y_right, impurity_fn):
    """
    Compute information gain from a split.

    Args:
        y: Parent node labels.
        y_left: Left child labels.
        y_right: Right child labels.
        impurity_fn: Function to compute impurity (gini_impurity or entropy).

    Returns:
        The information gain value.
    """
    parent = impurity_fn(y)
    n = len(y)
    return parent - (
        len(y_left) / n * impurity_fn(y_left)
        + len(y_right) / n * impurity_fn(y_right)
    )


# =====================================================================
# Decision Tree Classifier
# =====================================================================
class DecisionTreeClassifier(BaseDecisionTree):
    """
    Decision Tree Classifier for classification tasks.

    Uses impurity-based splits (Gini or Entropy) to recursively partition
    the feature space and create a tree for classification.

    Attributes:
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum samples required to split a node.
        impurity: Type of impurity metric ("gini" or "entropy").
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, impurity: str = "gini"):
        """
        Initialize the classifier.

        Args:
            max_depth: Maximum depth of the tree (default: 5).
            min_samples_split: Minimum samples to split (default: 2).
            impurity: Impurity metric ("gini" or "entropy", default: "gini").
        """
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        self.impurity = impurity
        self.impurity_fn = gini_impurity if impurity == "gini" else entropy

    # =====================================================================
    # Template Method Implementations
    # =====================================================================

    def _compute_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute information gain for a split."""
        return information_gain(y, y_left, y_right, self.impurity_fn)

    def _compute_leaf_value(self, y: np.ndarray) -> float:
        """Compute leaf value (most common class)."""
        return float(int(np.bincount(y).argmax()))

    def _should_stop_split(self, y: np.ndarray) -> bool:
        """Stop if all samples are from the same class."""
        return len(np.unique(y)) == 1

    def _compute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute misclassification error."""
        return float(np.mean(y_pred != y_true))

    def _get_cv_comparison_fn(self) -> Callable[[float, float], bool]:
        """For classification, higher accuracy is better."""
        return lambda new, best: new > best

    def _compute_cv_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(y_pred == y_true))
