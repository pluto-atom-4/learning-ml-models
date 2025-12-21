import numpy as np
from src.machine_learning.model_selection_polynomial import (
    mse,
    find_best_polynomial_degree,
)

def test_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])
    assert mse(y_true, y_pred) == 1/3


def test_find_best_polynomial_degree():
    # True function: y = 2 + 3x + x^2
    x = np.linspace(-3, 3, 50)
    y = 2 + 3*x + x**2

    # Split into train/validation
    x_train, x_val = x[:30], x[30:]
    y_train, y_val = y[:30], y[30:]

    train_errors, val_errors, best_degree = find_best_polynomial_degree(
        x_train, y_train, x_val, y_val, max_degree=5
    )

    assert len(train_errors) == 6
    assert len(val_errors) == 6
    assert best_degree == 2
