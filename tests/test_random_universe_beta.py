import pandas as pd
import numpy as np
from src.machine_learning.random_universe_beta import (
    compute_beta_coefficients,
    simulate_parallel_universes,
    compute_confidence_interval,
)


def mock_random_universe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic mock for testing.
    """
    df_new = df.copy()
    df_new["tv"] = df["tv"] + 0.1
    df_new["sales"] = df["sales"] + 0.2
    return df_new


def test_compute_beta_coefficients():
    df = pd.DataFrame({
        "tv": [1, 2, 3],
        "sales": [2, 4, 6],
    })

    beta0, beta1 = compute_beta_coefficients(df)

    assert np.isclose(beta1, 2.0)
    assert np.isclose(beta0, 0.0)


def test_simulate_parallel_universes():
    df = pd.DataFrame({
        "tv": [10, 20, 30],
        "sales": [15, 25, 35],
    })

    beta0_list, beta1_list = simulate_parallel_universes(
        df,
        RandomUniverse=mock_random_universe,
        n_universes=10,
    )

    assert len(beta0_list) == 10
    assert len(beta1_list) == 10


def test_compute_confidence_interval():
    values = [1, 2, 3, 4, 5]
    low, high = compute_confidence_interval(values, ci=80)

    assert low < high
    assert isinstance(low, float)
    assert isinstance(high, float)
