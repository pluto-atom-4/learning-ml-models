import numpy as np

from machine_learning.lasso_regression import LassoRegression


def test_lasso_regression_basic():
    X = [[1], [2], [3], [4]]
    y = [2, 4, 6, 8]

    model = LassoRegression(alpha=0.1, max_iter=5000)
    model.fit(X, y)

    preds = model.predict([[5], [6]])
    assert np.isclose(preds[0], 10, atol=2.2)
    assert np.isclose(preds[1], 12, atol=2.2)
