from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_land_data(csv_path: str | bytes | None):
    df = pd.read_csv(csv_path)
    X = df[["latitude", "longitude"]].to_numpy()
    y = df["land_type"].to_numpy().astype(int)
    return X, y


def fit_poly_logistic(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    C: float,
) -> tuple[LogisticRegression, PolynomialFeatures]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LogisticRegression(C=C, solver="lbfgs")
    model.fit(X_poly, y)

    return model, poly


def evaluate_model(model, poly, X, y):
    X_poly = poly.transform(X)
    preds = model.predict(X_poly)
    return accuracy_score(y, preds)


def search_best_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    degrees: list[int],
    C_values: list[float],
):
    """
    Iterate over combinations of polynomial degree and regularization parameter.
    Returns a list of (degree, C, accuracy, model, poly).
    """
    results = []

    for d in degrees:
        for C in C_values:
            model, poly = fit_poly_logistic(X, y, degree=d, C=C)
            acc = evaluate_model(model, poly, X, y)
            results.append((d, C, acc, model, poly))

    # Sort by accuracy descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results
