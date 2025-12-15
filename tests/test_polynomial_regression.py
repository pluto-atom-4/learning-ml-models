import numpy as np
from src.machine_learning.polynomial_regression import (
    expand_polynomial_features,
    fit_polynomial_regression,
    predict_polynomial_regression,
)

def test_expand_polynomial_features():
    x = np.array([1, 2, 3])
    features = expand_polynomial_features(x, degree=2)
    expected = np.array([
        [1, 1, 1],
        [1, 2, 4],
        [1, 3, 9],
    ])
    assert np.allclose(features, expected)


def test_fit_and_predict_polynomial_regression():
    # True relationship: y = 2 + 3x + x^2
    x = np.linspace(-3, 3, 20)
    y = 2 + 3*x + x**2

    weights = fit_polynomial_regression(x, y, degree=2)
    y_pred = predict_polynomial_regression(x, weights)

    assert np.allclose(y_pred, y, atol=1e-6)


def test_predict_shape():
    x = np.array([0, 1, 2])
    weights = np.array([1, 2, 3])  # y = 1 + 2x + 3x^2
    y_pred = predict_polynomial_regression(x, weights)

    assert y_pred.shape == x.shape
