import numpy as np

def expand_polynomial_features(x, degree):
    """
    Create polynomial features up to the given degree.
    Pure functional: no mutation.
    """
    x = np.asarray(x)
    return np.column_stack([x**i for i in range(degree + 1)])


def fit_polynomial_regression(x, y, degree):
    """
    Fit polynomial regression using the normal equation.
    Returns the weight vector.
    """
    X = expand_polynomial_features(x, degree)
    # Normal equation: w = (X^T X)^(-1) X^T y
    w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return w


def predict_polynomial_regression(x, weights):
    """
    Predict using learned polynomial regression weights.
    """
    degree = len(weights) - 1
    X = expand_polynomial_features(x, degree)
    return X @ weights
