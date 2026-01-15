"""
Unified Tree Node for Decision Trees and Regression Trees.

This module provides a single TreeNode class used by both
classification and regression tree implementations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    """
    A node in a decision or regression tree.

    Attributes:
        feature_index: Index of the feature used for splitting. None for leaf nodes.
        threshold: Threshold value for the split. None for leaf nodes.
        left: Left child node (samples <= threshold).
        right: Right child node (samples > threshold).
        value: Predicted value (int for classification, float for regression).
               None for internal nodes.
    """
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[float] = None  # Works for both int and float

