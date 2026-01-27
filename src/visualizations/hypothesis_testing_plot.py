from __future__ import annotations
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from machine_learning.hypothesis_testing import (
    fit_linear_regression,
    bootstrap_coefficients,
    compute_t_values,
    compute_p_values_from_t,
)


def plot_coefficients(df: pd.DataFrame, response_col: str) -> None:
    res = fit_linear_regression(df, response_col=response_col)
    y = np.arange(len(res.feature_names))

    plt.figure(figsize=(8, 5))
    plt.barh(y, res.coefficients)
    plt.yticks(y, res.feature_names)
    plt.xlabel("Coefficient value")
    plt.title("Linear Regression Coefficients")
    plt.tight_layout()
    plt.show()


def plot_t_values(
    df: pd.DataFrame,
    response_col: str,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> None:
    feature_names, coef_samples, n_samples = bootstrap_coefficients(
        df, response_col=response_col, n_bootstrap=n_bootstrap, random_state=random_state
    )
    t_vals = compute_t_values(coef_samples)

    y = np.arange(len(feature_names))
    plt.figure(figsize=(8, 5))
    plt.barh(y, t_vals)
    plt.yticks(y, feature_names)
    plt.xlabel("|t| value")
    plt.title("Bootstrap |t| Values per Coefficient")
    plt.tight_layout()
    plt.show()


def plot_one_minus_p(
    df: pd.DataFrame,
    response_col: str,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> None:
    feature_names, coef_samples, n_samples = bootstrap_coefficients(
        df, response_col=response_col, n_bootstrap=n_bootstrap, random_state=random_state
    )
    t_vals = compute_t_values(coef_samples)
    dof = n_samples - (coef_samples.shape[1] - 1) - 1
    p_vals = compute_p_values_from_t(t_vals, dof=dof)
    one_minus_p = 1 - p_vals

    y = np.arange(len(feature_names))
    plt.figure(figsize=(8, 5))
    plt.barh(y, one_minus_p)
    plt.axvline(0.95, color="red", linestyle="--", label="0.95 threshold")
    plt.yticks(y, feature_names)
    plt.xlabel("1 - p value")
    plt.title("Hypothesis Testing: 1 - p per Coefficient")
    plt.legend()
    plt.tight_layout()
    plt.show()