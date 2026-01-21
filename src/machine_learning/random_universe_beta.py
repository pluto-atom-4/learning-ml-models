from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Tuple, List


def compute_beta_coefficients(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute beta0 and beta1 for simple linear regression:
        sales ~ beta0 + beta1 * tv
    """
    x = df["tv"].to_numpy()
    y = df["sales"].to_numpy()

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        raise ValueError("Variance of x is zero; cannot compute beta1")

    beta1 = numerator / denominator
    beta0 = y_mean - beta1 * x_mean

    return float(beta0), float(beta1)


def simulate_parallel_universes(
    df: pd.DataFrame,
    RandomUniverse: Callable[[pd.DataFrame], pd.DataFrame],
    n_universes: int = 100,
) -> Tuple[List[float], List[float]]:
    """
    Generate beta0 and beta1 values across many parallel universes.
    """
    beta0_list = []
    beta1_list = []

    for _ in range(n_universes):
        df_new = RandomUniverse(df)
        beta0, beta1 = compute_beta_coefficients(df_new)
        beta0_list.append(beta0)
        beta1_list.append(beta1)

    return beta0_list, beta1_list


def compute_confidence_interval(values: List[float], ci: float = 95) -> Tuple[float, float]:
    """
    Compute a confidence interval using percentiles.
    """
    lower = (100 - ci) / 2
    upper = 100 - lower
    return (
        float(np.percentile(values, lower)),
        float(np.percentile(values, upper)),
    )
