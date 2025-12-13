import numpy as np

from machine_learning.linear_regression import LinearRegression


def test_linear_regression_fit_predict():
    X = [[1], [2], [3], [4]]
    y = [2, 4, 6, 8]  # Perfect linear relationship: y = 2x

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict([[5], [6]])
    assert np.isclose(preds[0], 10)
    assert np.isclose(preds[1], 12)

def test_linear_regression_not_fitted():
    model = LinearRegression()
    try:
        model.predict([[1]])
        assert False  # Should not reach here
    except ValueError:
        assert True
