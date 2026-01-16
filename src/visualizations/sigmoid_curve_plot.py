from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray, L: float = 1.0, k: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """
    Standard sigmoid function:
        S(x) = L / (1 + exp(-k(x - x0)))

    Parameters
    ----------
    x : np.ndarray
        Input values.
    L : float
        Maximum value of the curve.
    k : float
        Steepness of the curve.
    x0 : float
        Midpoint of the curve.

    Returns
    -------
    np.ndarray
        Sigmoid-transformed values.
    """
    return L / (1 + np.exp(-k * (x - x0)))


def plot_sigmoid(
    x_range: tuple[float, float] = (-10, 10),
    L: float = 1.0,
    k: float = 1.0,
    x0: float = 0.0,
    num_points: int = 400,
    title: str = "Sigmoid Curve",
) -> None:
    """
    Plot a sigmoid curve over a specified range.

    Parameters
    ----------
    x_range : tuple
        Range of x-values (min, max).
    L : float
        Maximum value of the curve.
    k : float
        Steepness.
    x0 : float
        Midpoint.
    num_points : int
        Number of points to sample.
    title : str
        Plot title.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = sigmoid(x, L=L, k=k, x0=x0)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"L={L}, k={k}, x0={x0}", linewidth=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Sigmoid(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
