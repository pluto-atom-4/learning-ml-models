from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable
from machine_learning.random_universe_beta import (
    compute_beta_coefficients,
    simulate_parallel_universes,
)


def RandomUniverse(df):
    df_bootstrap = df.sample(200, replace = True)
    return df_bootstrap

def plot_beta_distributions(
    df: pd.DataFrame,
    RandomUniverse: Callable[[pd.DataFrame], pd.DataFrame],
    n_universes: int = 200,
) -> None:
    """
    Generate parallel-universe beta coefficients and plot histograms.
    """
    beta0_list, beta1_list = simulate_parallel_universes(
        df,
        RandomUniverse=RandomUniverse,
        n_universes=n_universes,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(beta0_list, bins=30, color="steelblue", edgecolor="black")
    axes[0].set_title(f"Distribution of β₀ across {n_universes} universes")
    axes[0].set_xlabel("β₀")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(beta1_list, bins=30, color="darkorange", edgecolor="black")
    axes[1].set_title(f"Distribution of β₁ across {n_universes} universes")
    axes[1].set_xlabel("β₁")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_original_vs_parallel(
    df: pd.DataFrame,
    RandomUniverse: Callable[[pd.DataFrame], pd.DataFrame],
    n_samples: int = 5,
) -> None:
    """
    Plot the original dataset and a few parallel-universe datasets.
    """
    plt.figure(figsize=(8, 6))

    # Original dataset
    plt.scatter(df["tv"], df["sales"], label="Original", color="black", s=60)

    # Parallel universes
    for i in range(n_samples):
        df_new = RandomUniverse(df)
        plt.scatter(
            df_new["tv"],
            df_new["sales"],
            alpha=0.5,
            s=40,
            label=f"Universe {i+1}",
        )

    plt.title("Original vs Parallel-Universe Datasets")
    plt.xlabel("TV Budget")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
