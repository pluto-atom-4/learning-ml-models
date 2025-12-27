import matplotlib.pyplot as plt
import numpy as np


def plot_bias_variance(x_sample, y_sample, x_pred, predictions, true_f=None, save_path=None):
    """
    Plot:
      - sampled noisy points
      - true function (optional)
      - predicted curves for each degree
    """
    plt.figure(figsize=(10, 6))

    # Plot sampled noisy data
    plt.scatter(x_sample, y_sample, color="black", label="Sampled Data", alpha=0.7)

    # Plot true function if provided
    if true_f is not None:
        plt.plot(x_pred, true_f, color="green", linewidth=2, label="True Function")

    # Plot predicted curves
    for degree, y_pred in predictions.items():
        plt.plot(x_pred, y_pred, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Bias–Variance Tradeoff: Polynomial Regression")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
