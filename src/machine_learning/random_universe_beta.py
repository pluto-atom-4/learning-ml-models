from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Tuple, List


def compute_beta_coefficients(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute beta0 and beta1 for simple linear regression:
        sales ~ beta0 + beta1 * tv
    using the closed-form OLS equations.
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
    Repeatedly:
      1. Generate a new dataset using RandomUniverse(df)
      2. Compute beta0, beta1
      3. Store results

    Returns:
        beta0_list, beta1_list
    """
    beta0_list = []
    beta1_list = []

    for _ in range(n_universes):
        df_new = RandomUniverse(df)
        beta0, beta1 = compute_beta_coefficients(df_new)
        beta0_list.append(beta0)
        beta1_list.append(beta1)

    return beta0_list, beta1_list
