from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable
from src.machine_learning.random_universe_beta import (
    simulate_parallel_universes,
    compute_confidence_interval,
)

def randomUniverse(df):
    df_bootstrap = df.sample(200, replace = True)
    return df_bootstrap


def plot_simulation(values, ci_low, ci_high, title):
    """
    Helper plot: histogram of beta distribution with CI band.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, alpha=0.7, color="steelblue", edgecolor="black", label="β distribution")
    plt.axvline(ci_low, color="red", linestyle="--", linewidth=2, label="Lower CI")
    plt.axvline(ci_high, color="green", linestyle="--", linewidth=2, label="Upper CI")
    plt.title("95% Confidence Interval")
    plt.xlabel(r"$\beta$ value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.show()


def plot_beta_confidence_intervals(
    df: pd.DataFrame,
    RandomUniverse: Callable[[pd.DataFrame], pd.DataFrame],
    n_universes: int = 200,
):
    beta0_list, beta1_list = simulate_parallel_universes(
        df,
        RandomUniverse=RandomUniverse,
        n_universes=n_universes,
    )

    beta0_low, beta0_high = compute_confidence_interval(beta0_list)
    beta1_low, beta1_high = compute_confidence_interval(beta1_list)

    plot_simulation(beta0_list, beta0_low, beta0_high, "β₀ Confidence Interval")
    plot_simulation(beta1_list, beta1_low, beta1_high, "β₁ Confidence Interval")
