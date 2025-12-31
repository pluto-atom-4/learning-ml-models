from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def logistic(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def plot_data_only(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Insurance Claim vs Age",
    save_path: str | Path | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="black", alpha=0.7, label="Observed Data")

    plt.xlabel("Age")
    plt.ylabel("Insurance Claim (0/1)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()


def plot_with_logistic_curve(
    x: np.ndarray,
    y: np.ndarray,
    beta0: float,
    beta1: float,
    title: str = "Logistic Fit Curve",
    save_path: str | Path | None = None,
) -> None:

    x_sorted = np.linspace(x.min(), x.max(), 200)
    probs = logistic(beta0 + beta1 * x_sorted)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="black", alpha=0.7, label="Observed Data")
    plt.plot(x_sorted, probs, color="red", linewidth=2, label="Logistic Curve")

    plt.xlabel("Age")
    plt.ylabel("Probability of Insurance Claim")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
