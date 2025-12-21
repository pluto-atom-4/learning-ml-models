import numpy as np
from src.machine_learning.polynomial_regression import (
    fit_polynomial_regression,
    predict_polynomial_regression,
)

def mse(y_true, y_pred):
    """Mean squared error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def find_best_polynomial_degree(x_train, y_train, x_val, y_val, max_degree):
    """
    Evaluate polynomial regression models from degree 0..max_degree.
    Returns:
        train_errors: list of MSE values for training data
        val_errors: list of MSE values for validation data
        best_degree: degree with lowest validation MSE
    """
    train_errors = []
    val_errors = []

    for degree in range(max_degree + 1):
        weights = fit_polynomial_regression(x_train, y_train, degree)

        y_train_pred = predict_polynomial_regression(x_train, weights)
        y_val_pred = predict_polynomial_regression(x_val, weights)

        train_errors.append(mse(y_train, y_train_pred))
        val_errors.append(mse(y_val, y_val_pred))

    best_degree = int(np.argmin(val_errors))
    return train_errors, val_errors, best_degree
