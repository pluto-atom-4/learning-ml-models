import numpy as np
from machine_learning.decision_tree import (DecisionTreeClassifier, gini_impurity, entropy)


def test_gini_impurity():
    y = np.array([0, 0, 1, 1])
    assert gini_impurity(y) == 0.5


def test_entropy():
    y = np.array([0, 0, 1, 1])
    assert round(entropy(y), 3) == 1.0


def test_tree_fit_predict():
    X = np.array([[1], [2], [10], [11]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, impurity="gini")
    tree.fit(X, y)

    preds = tree.predict(X)
    assert np.array_equal(preds, y)
