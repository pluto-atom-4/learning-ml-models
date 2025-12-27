import numpy as np
import pandas as pd
from src.machine_learning.polynomial_regression import (
    fit_polynomial_regression,
    predict_polynomial_regression,
)


def load_noisy_population(csv_path):
    """
    Load dataset with columns f, x, y.
    Returns numpy arrays f, x, y.
    """
    df = pd.read_csv(csv_path)
    return df["f"].values, df["x"].values, df["y"].values


def sample_dataset(x, y, sample_size, random_state=None):
    """
    Randomly sample a subset of the dataset.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(x), size=sample_size, replace=False)
    return x[idx], y[idx]


def compute_polynomial_predictions(x_sample, y_sample, degrees, num_points=100):
    """
    For each degree:
      - Fit polynomial regression
      - Predict on random points
    Returns:
      predictions: dict degree -> predicted y-values
      x_pred: the x-values used for prediction
    """
    rng = np.random.default_rng()
    x_pred = np.sort(rng.uniform(low=min(x_sample), high=max(x_sample), size=num_points))

    predictions = {}

    for degree in degrees:
        weights = fit_polynomial_regression(x_sample, y_sample, degree)
        y_pred = predict_polynomial_regression(x_pred, weights)
        predictions[degree] = y_pred

    return predictions, x_pred
