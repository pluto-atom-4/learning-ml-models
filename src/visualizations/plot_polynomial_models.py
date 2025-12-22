import numpy as np
import matplotlib.pyplot as plt
import csv
from src.machine_learning.polynomial_regression import (
    fit_polynomial_regression,
    predict_polynomial_regression,
)


def load_dataset(csv_path):
    """
    Load x,y dataset from CSV file.
    Returns numpy arrays x, y.
    """
    xs, ys = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return np.array(xs), np.array(ys)


def plot_polynomial_models(x, y, max_degree, save_path=None):
    """
    Plot the dataset and multiple polynomial regression model curves.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot of dataset
    plt.scatter(x, y, color="black", label="Data Points")

    # Smooth x-range for plotting curves
    x_plot = np.linspace(min(x), max(x), 300)

    # Plot each polynomial model
    for degree in range(max_degree + 1):
        weights = fit_polynomial_regression(x, y, degree)
        y_plot = predict_polynomial_regression(x_plot, weights)
        plt.plot(x_plot, y_plot, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial Regression Models")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
