import numpy as np
from src.machine_learning.multi_linear_regression import (
    add_bias_column,
    fit_multi_linear_regression,
    predict_multi_linear_regression,
)

def test_add_bias_column():
    X = np.array([[1, 2], [3, 4]])
    out = add_bias_column(X)
    expected = np.array([[1, 1, 2],
                         [1, 3, 4]])
    assert np.allclose(out, expected)


def test_fit_and_predict_multi_linear_regression():
    # True model: y = 2 + 3*x1 + 4*x2
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 0],
        [0, 3],
    ])
    y = 2 + 3*X[:, 0] + 4*X[:, 1]

    w = fit_multi_linear_regression(X, y)
    y_pred = predict_multi_linear_regression(X, w)

    assert np.allclose(y_pred, y, atol=1e-6)


def test_predict_shape():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    weights = np.array([1, 2, 3])  # y = 1 + 2*x1 + 3*x2
    y_pred = predict_multi_linear_regression(X, weights)

    assert y_pred.shape == (3,)
