from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import t as student_t


@dataclass
class RegressionResult:
    feature_names: List[str]
    coefficients: np.ndarray  # includes intercept as first element
    n_samples: int


def fit_linear_regression(
    df: pd.DataFrame,
    response_col: str,
) -> RegressionResult:
    """Fit multi-linear regression with MinMax scaling on predictors."""
    X = df.drop(columns=[response_col]).to_numpy(dtype=float)
    y = df[response_col].to_numpy(dtype=float)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    coef = np.concatenate(([model.intercept_], model.coef_))
    feature_names = ["intercept"] + list(df.drop(columns=[response_col]).columns)

    return RegressionResult(feature_names=feature_names, coefficients=coef, n_samples=len(df))


def bootstrap_coefficients(
    df: pd.DataFrame,
    response_col: str,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> Tuple[List[str], np.ndarray, int]:
    """Bootstrap regression coefficients across many resamples."""
    rng = np.random.default_rng(random_state)
    n = len(df)

    base_result = fit_linear_regression(df, response_col)
    feature_names = base_result.feature_names
    n_features = len(feature_names)

    coef_samples = np.zeros((n_bootstrap, n_features))

    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        df_sample = df.iloc[sample_idx]
        res = fit_linear_regression(df_sample, response_col)
        coef_samples[i, :] = res.coefficients

    return feature_names, coef_samples, base_result.n_samples


def compute_t_values(coef_samples: np.ndarray) -> np.ndarray:
    """
    Compute |t| values per coefficient using bootstrap:
    t_j = |mean_j / std_j|.
    """
    means = coef_samples.mean(axis=0)
    stds = coef_samples.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_vals = np.abs(means / stds)
        t_vals[np.isnan(t_vals)] = 0.0
        t_vals[np.isinf(t_vals)] = 0.0
    return t_vals


def compute_p_values_from_t(
    t_values: np.ndarray,
    dof: int,
) -> np.ndarray:
    """Two-sided p-values from |t| values."""
    # p = 2 * (1 - CDF(|t|))
    p = 2 * (1 - student_t.cdf(t_values, df=dof))
    return p


def compute_percentiles(
    values: Sequence[float] | np.ndarray,
    percentiles: Sequence[float],
) -> np.ndarray:
    """Compute n-th percentiles for a sequence of values."""
    return np.percentile(values, percentiles)
