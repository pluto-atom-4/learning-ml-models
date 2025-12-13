import numpy as np

from machine_learning.ridge_regression import RidgeRegression


def test_ridge_regression_basic():
    X = [[1], [2], [3], [4]]
    y = [2, 4, 6, 8]

    model = RidgeRegression(alpha=1.0)
    model.fit(X, y)

    preds = model.predict([[5], [6]])
    assert np.isclose(preds[0], 10, atol=1.5)
    assert np.isclose(preds[1], 12, atol=1.5)
