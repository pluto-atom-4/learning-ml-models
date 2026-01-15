"""
Base Decision Tree class with shared functionality.

This module provides the abstract base class for both classification
and regression tree implementations, extracting common patterns and
allowing subclasses to specialize for their task.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Callable, Tuple, List

from .tree_node import TreeNode


class BaseDecisionTree(ABC):
    """
    Abstract base class for decision trees (classification and regression).

    Handles common tree operations: fitting, predicting, pruning, and
    cross-validation. Subclasses define task-specific behavior through
    abstract methods.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        gain_fn: Optional[Callable] = None,
    ):
        """
        Initialize the tree.

        Args:
            max_depth: Maximum depth of the tree.
            min_samples_split: Minimum samples required to split a node.
            gain_fn: Function to compute gain for splitting. If None,
                     subclass must implement _compute_gain().
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[TreeNode] = None
        self._gain_fn = gain_fn

    # =====================================================================
    # Core Tree Operations
    # =====================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the tree to training data."""
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on samples."""
        return np.array([self._predict_one(x, self.root) for x in X])

    # =====================================================================
    # Template Methods (To be implemented by subclasses)
    # =====================================================================

    @abstractmethod
    def _compute_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Compute the gain (e.g., information gain or variance reduction) for a split.

        Args:
            y: Target values of parent node.
            y_left: Target values of left child.
            y_right: Target values of right child.

        Returns:
            The gain value (higher is better).
        """
        pass

    @abstractmethod
    def _compute_leaf_value(self, y: np.ndarray) -> float:
        """
        Compute the leaf node value (e.g., mode for classification, mean for regression).

        Args:
            y: Target values for the node.

        Returns:
            The leaf value as a float.
        """
        pass

    @abstractmethod
    def _should_stop_split(self, y: np.ndarray) -> bool:
        """
        Determine if a node should be split (stopping criterion).

        Args:
            y: Target values for the node.

        Returns:
            True if the node should not be split further.
        """
        pass

    @abstractmethod
    def _compute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute error metric for pruning and CV (e.g., misclassification or MSE).

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            The error value.
        """
        pass

    @abstractmethod
    def _get_cv_comparison_fn(self) -> Callable[[float, float], bool]:
        """
        Get the comparison function for CV score selection.

        Returns:
            A function that returns True if the new score is better.
            For classification: lambda new, best: new > best
            For regression: lambda new, best: new < best
        """
        pass

    # =====================================================================
    # Best Split Logic (Common across all trees)
    # =====================================================================

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split for a node.

        Returns:
            Tuple of (best_feature, best_threshold, best_gain).
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                gain = self._compute_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold, best_gain

    # =====================================================================
    # Tree Building (Common structure, task-specific logic)
    # =====================================================================

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        """
        Recursively build the tree.

        Args:
            X: Feature matrix.
            y: Target values.
            depth: Current depth (for max_depth check).

        Returns:
            The root node of the constructed subtree.
        """
        # Check stopping criteria
        if depth >= self.max_depth or self._should_stop_split(y):
            return TreeNode(value=self._compute_leaf_value(y))

        # Find best split
        feature, threshold, gain = self._best_split(X, y)

        # No good split found
        if gain <= 0 or feature is None:
            return TreeNode(value=self._compute_leaf_value(y))

        # Perform split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return TreeNode(
            feature_index=feature,
            threshold=threshold,
            left=self._build_tree(X[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(X[right_mask], y[right_mask], depth + 1),
        )

    # =====================================================================
    # Prediction (Common across all trees)
    # =====================================================================

    def _predict_one(self, x: np.ndarray, node: Optional[TreeNode]) -> float:
        """Make prediction for a single sample."""
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    # =====================================================================
    # Pruning (Common algorithm with task-specific error metrics)
    # =====================================================================

    def _subtree_leaves(self, node: Optional[TreeNode]) -> int:
        """Count number of leaves in a subtree."""
        if node.value is not None:
            return 1
        return self._subtree_leaves(node.left) + self._subtree_leaves(node.right)

    def _subtree_error(self, X: np.ndarray, y: np.ndarray, node: Optional[TreeNode]) -> float:
        """Compute error for a subtree."""
        preds = np.array([self._predict_one(x, node) for x in X])
        return self._compute_error(y, preds)

    def prune(self, X: np.ndarray, y: np.ndarray, alpha: float) -> None:
        """
        Prune the tree using cost-complexity pruning.

        Args:
            X: Feature matrix.
            y: Target values.
            alpha: Complexity parameter (higher = more pruning).
        """
        def prune_node(node: TreeNode) -> TreeNode:
            if node.value is not None:
                return node

            node.left = prune_node(node.left)
            node.right = prune_node(node.right)

            # Compute cost of subtree
            leaves = self._subtree_leaves(node)
            subtree_cost = self._subtree_error(X, y, node) + alpha * leaves

            # Compute cost of replacing with leaf
            leaf_value = self._compute_leaf_value(y)
            leaf_pred = np.full_like(y, fill_value=leaf_value, dtype=float)
            leaf_cost = self._compute_error(y, leaf_pred) + alpha * 1

            if leaf_cost <= subtree_cost:
                return TreeNode(value=leaf_value)

            return node

        self.root = prune_node(self.root)

    # =====================================================================
    # Cross-Validation (Common structure with task-specific scoring)
    # =====================================================================

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depths: List[int],
        min_samples: List[int],
        k: int = 5,
    ) -> Tuple[Tuple[int, int], float]:
        """
        Perform k-fold cross-validation to find optimal hyperparameters.

        Args:
            X: Feature matrix.
            y: Target values.
            depths: List of max_depth values to try.
            min_samples: List of min_samples_split values to try.
            k: Number of folds.

        Returns:
            Tuple of (best_params, best_score) where best_params = (depth, min_samples).
        """
        comparison_fn = self._get_cv_comparison_fn()
        # For classification (higher is better): start at -inf
        # For regression (lower is better): start at +inf
        best_score = float("-inf") if comparison_fn(0, float("-inf")) else float("inf")
        best_params = None

        n = len(X)
        k = min(k, n)  # Adjust k if dataset is smaller

        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)

        for d in depths:
            for m in min_samples:
                scores = []
                for i in range(k):
                    val_idx = folds[i]
                    train_idx = np.hstack([folds[j] for j in range(k) if j != i])

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    model = self.__class__(max_depth=d, min_samples_split=m)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    scores.append(self._compute_cv_score(y_val, preds))

                avg_score = float(np.mean(scores))

                if comparison_fn(avg_score, best_score):
                    best_score = avg_score
                    best_params = (d, m)

        return best_params, best_score

    @abstractmethod
    def _compute_cv_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute CV score for a fold (accuracy for classification, MSE for regression).

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            The score for this fold.
        """
        pass

