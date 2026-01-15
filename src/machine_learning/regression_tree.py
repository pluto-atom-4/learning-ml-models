"""
Regression Tree for regression tasks.

Implements a regression tree using variance reduction (MSE-based splits).
Inherits common tree functionality from BaseDecisionTree.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Callable

from .base_tree import BaseDecisionTree


# =====================================================================
# Regression Metrics (Regression-specific)
# =====================================================================
def mse(y: np.ndarray) -> float:
    """Compute mean squared error (MSE) for a set of values."""
    return float(np.mean((y - np.mean(y)) ** 2))


def variance_reduction(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """
    Compute variance reduction from a split.

    Args:
        y: Parent node values.
        y_left: Left child values.
        y_right: Right child values.

    Returns:
        The variance reduction value.
    """
    parent = mse(y)
    n = len(y)
    return parent - (
        len(y_left) / n * mse(y_left)
        + len(y_right) / n * mse(y_right)
    )


# =====================================================================
# Regression Tree
# =====================================================================
class RegressionTree(BaseDecisionTree):
    """
    Regression Tree for regression tasks.

    Uses variance reduction (MSE-based) splits to recursively partition
    the feature space and create a tree for regression.

    Attributes:
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum samples required to split a node.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        """
        Initialize the regression tree.

        Args:
            max_depth: Maximum depth of the tree (default: 5).
            min_samples_split: Minimum samples to split (default: 2).
        """
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)

    # =====================================================================
    # Template Method Implementations
    # =====================================================================

    def _compute_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute variance reduction for a split."""
        return variance_reduction(y, y_left, y_right)

    def _compute_leaf_value(self, y: np.ndarray) -> float:
        """Compute leaf value (mean of target values)."""
        return float(np.mean(y))

    def _should_stop_split(self, y: np.ndarray) -> bool:
        """Stop if fewer than min_samples_split remain."""
        return len(y) < self.min_samples_split

    def _compute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error."""
        return float(np.mean((y_pred - y_true) ** 2))

    def _get_cv_comparison_fn(self) -> Callable[[float, float], bool]:
        """For regression, lower MSE is better."""
        return lambda new, best: new < best

    def _compute_cv_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error for CV."""
        return float(np.mean((y_pred - y_true) ** 2))

