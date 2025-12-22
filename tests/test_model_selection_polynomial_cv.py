import numpy as np
from src.machine_learning.model_selection_polynomial_cv import (
    mse,
    find_best_degree_with_cv,
)

def test_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])
    assert mse(y_true, y_pred) == 1/3


def test_find_best_degree_with_cv():
    # True function: y = 2 + 3x + x^2
    x = np.linspace(-3, 3, 60)
    y = 2 + 3*x + x**2

    # Split into train/validation
    x_train, x_val = x[:40], x[40:]
    y_train, y_val = y[:40], y[40:]

    (
        train_errors,
        val_errors,
        cv_errors,
        best_val_degree,
        best_cv_degree,
    ) = find_best_degree_with_cv(
        x_train, y_train, x_val, y_val, max_degree=6, k=4
    )

    assert len(train_errors) == 7
    assert len(val_errors) == 7
    assert len(cv_errors) == 7

    # True polynomial degree is 2
    assert best_val_degree == 2
    assert best_cv_degree == 2
