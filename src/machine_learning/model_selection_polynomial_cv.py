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


def k_fold_split(x, y, k):
    """Pure functional k-fold splitter."""
    x = np.asarray(x)
    y = np.asarray(y)
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    fold_sizes = np.full(k, len(x) // k)
    fold_sizes[: len(x) % k] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop

    return folds


def find_best_degree_with_cv(x_train, y_train, x_val, y_val, max_degree, k=5):
    """
    Returns:
        train_errors: list of MSE values for training data
        val_errors: list of MSE values for validation data
        cv_errors: list of cross-validation MSE values
        best_val_degree: degree with lowest validation MSE
        best_cv_degree: degree with lowest CV MSE
    """
    train_errors = []
    val_errors = []
    cv_errors = []

    folds = k_fold_split(x_train, y_train, k)

    for degree in range(max_degree + 1):
        # Train/validation errors
        weights = fit_polynomial_regression(x_train, y_train, degree)
        y_train_pred = predict_polynomial_regression(x_train, weights)
        y_val_pred = predict_polynomial_regression(x_val, weights)

        train_errors.append(mse(y_train, y_train_pred))
        val_errors.append(mse(y_val, y_val_pred))

        # Cross-validation
        fold_mse = []
        for train_idx, val_idx in folds:
            x_tr, y_tr = x_train[train_idx], y_train[train_idx]
            x_vl, y_vl = x_train[val_idx], y_train[val_idx]

            w = fit_polynomial_regression(x_tr, y_tr, degree)
            y_pred = predict_polynomial_regression(x_vl, w)
            fold_mse.append(mse(y_vl, y_pred))

        cv_errors.append(np.mean(fold_mse))

    best_val_degree = int(np.argmin(val_errors))
    best_cv_degree = int(np.argmin(cv_errors))

    return (
        train_errors,
        val_errors,
        cv_errors,
        best_val_degree,
        best_cv_degree,
    )
