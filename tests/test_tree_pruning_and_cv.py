import numpy as np
from machine_learning.decision_tree import DecisionTreeClassifier
from machine_learning.regression_tree import RegressionTree


def test_classification_cv():
    X = np.array([[1], [2], [10], [11]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier()
    params, score = tree.cross_validate(X, y, depths=[1, 2, 3], min_samples=[1, 2])
    assert params is not None
    assert 0 <= score <= 1


def test_regression_cv():
    X = np.array([[1], [2], [10], [11]])
    y = np.array([1.0, 1.5, 10.0, 10.5])

    tree = RegressionTree()
    params, score = tree.cross_validate(X, y, depths=[1, 2, 3], min_samples=[1, 2])
    assert params is not None
    assert score >= 0


def test_pruning_classification():
    X = np.array([[1], [2], [10], [11]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X, y)
    tree.prune(X, y, alpha=0.1)

    preds = tree.predict(X)
    assert preds.shape == y.shape


def test_pruning_regression():
    X = np.array([[1], [2], [10], [11]])
    y = np.array([1.0, 1.5, 10.0, 10.5])

    tree = RegressionTree(max_depth=5)
    tree.fit(X, y)
    tree.prune(X, y, alpha=0.1)

    preds = tree.predict(X)
    assert preds.shape == y.shape
