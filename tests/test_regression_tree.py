import numpy as np
from machine_learning.regression_tree import (RegressionTree, mse, variance_reduction)


def test_mse():
    y = np.array([1, 1, 3, 3])
    assert mse(y) == 1.0


def test_variance_reduction():
    y = np.array([1, 1, 3, 3])
    y_left = np.array([1, 1])
    y_right = np.array([3, 3])

    gain = variance_reduction(y, y_left, y_right)
    assert gain > 0


def test_regression_tree_fit_predict():
    X = np.array([[1], [2], [10], [11]])
    y = np.array([1.0, 1.5, 10.0, 10.5])

    tree = RegressionTree(max_depth=3)
    tree.fit(X, y)

    preds = tree.predict(X)
    assert preds.shape == y.shape
    assert np.allclose(preds, y, atol=1.0)
