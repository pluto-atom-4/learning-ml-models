from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_advertising_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load TV advertising spending and Sales from CSV file.

    Args:
        csv_path: Path to Advertising-three-media.csv

    Returns:
        Tuple of (TV array, Sales array) as numpy arrays
    """
    tv_values = []
    sales_values = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tv_values.append(float(row["TV"]))
            sales_values.append(float(row["Sales"]))

    return np.array(tv_values), np.array(sales_values)


def plot_regression(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Linear Regression: TV Advertising vs Sales",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot actual data points and regression line.

    Args:
        x: Feature values (TV spending)
        y: Actual target values (Sales)
        y_pred: Predicted target values from the model
        title: Plot title
        save_path: Optional path to save the figure
    """
    # Sort by x for clean regression line
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.figure(figsize=(10, 6))

    # Scatter plot of actual data
    plt.scatter(x, y, color="black", alpha=0.6, s=50, label="Actual Data")

    # Regression line
    plt.plot(x_sorted, y_pred_sorted, color="red", linewidth=2, label="Regression Line")

    plt.xlabel("TV Advertising Spending ($1000s)")
    plt.ylabel("Sales ($1000s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
