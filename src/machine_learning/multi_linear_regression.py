import numpy as np

def add_bias_column(X):
    """
    Add a column of ones (bias term) to the feature matrix.
    Pure functional: returns a new matrix.
    """
    X = np.asarray(X)
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def fit_multi_linear_regression(X, y):
    """
    Fit multivariate linear regression using the normal equation.
    Returns weight vector w.
    """
    X = add_bias_column(X)
    y = np.asarray(y)

    # Normal equation: w = (X^T X)^(-1) X^T y
    w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return w


def predict_multi_linear_regression(X, weights):
    """
    Predict using learned weights.
    """
    X = add_bias_column(X)
    return X @ weights
