import numpy as np
import pandas as pd

from src.machine_learning.hypothesis_testing import (
    fit_linear_regression,
    bootstrap_coefficients,
    compute_t_values,
    compute_p_values_from_t,
    compute_percentiles,
)


def make_toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=100)
    X2 = rng.normal(size=100)
    y = 3.0 + 2.0 * X1 - 1.0 * X2 + rng.normal(scale=0.5, size=100)
    return pd.DataFrame({"X1": X1, "X2": X2, "medv": y})


def test_fit_linear_regression():
    df = make_toy_df()
    res = fit_linear_regression(df, response_col="medv")

    assert "intercept" in res.feature_names
    assert len(res.feature_names) == len(res.coefficients)
    assert res.n_samples == len(df)


def test_bootstrap_coefficients_shape():
    df = make_toy_df()
    feature_names, coef_samples, n_samples = bootstrap_coefficients(
        df, response_col="medv", n_bootstrap=50, random_state=42
    )

    assert coef_samples.shape == (50, len(feature_names))
    assert n_samples == len(df)


def test_compute_t_and_p_values():
    df = make_toy_df()
    feature_names, coef_samples, n_samples = bootstrap_coefficients(
        df, response_col="medv", n_bootstrap=100, random_state=1
    )

    t_vals = compute_t_values(coef_samples)
    assert t_vals.shape[0] == coef_samples.shape[1]
    assert np.all(t_vals >= 0)

    dof = n_samples - (coef_samples.shape[1] - 1) - 1
    p_vals = compute_p_values_from_t(t_vals, dof=dof)
    assert p_vals.shape == t_vals.shape
    assert np.all((p_vals >= 0) & (p_vals <= 1))


def test_compute_percentiles():
    vals = [1, 2, 3, 4, 5]
    perc = compute_percentiles(vals, [0, 50, 100])
    assert np.allclose(perc, [1, 3, 5])
