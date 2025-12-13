import numpy as np

from machine_learning.mse import mean_squared_error


def test_mse_basic():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert mean_squared_error(y_true, y_pred) == 0.0

def test_mse_nonzero():
    y_true = [1, 2, 3]
    y_pred = [2, 2, 2]
    assert np.isclose(mean_squared_error(y_true, y_pred), 2/3)
