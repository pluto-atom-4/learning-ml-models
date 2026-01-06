import numpy as np
import pandas as pd
from src.machine_learning.multi_poly_land_classifier import (
    load_land_data,
    fit_poly_logistic,
    evaluate_model,
    search_best_hyperparams,
)


def test_load_land_data(tmp_path):
    csv_file = tmp_path / "land.csv"
    csv_file.write_text(
        "latitude,longitude,land_type\n"
        "1,2,0\n"
        "3,4,1\n"
    )

    X, y = load_land_data(csv_file)
    assert X.shape == (2, 2)
    assert np.array_equal(y, np.array([0, 1]))


def test_fit_and_evaluate():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])

    model, poly = fit_poly_logistic(X, y, degree=2, C=1.0)
    acc = evaluate_model(model, poly, X, y)

    assert 0 <= acc <= 1


def test_search_best_hyperparams():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])

    results = search_best_hyperparams(X, y, degrees=[1, 2], C_values=[0.1, 1])
    assert len(results) == 4
    assert results[0][2] >= results[-1][2]
