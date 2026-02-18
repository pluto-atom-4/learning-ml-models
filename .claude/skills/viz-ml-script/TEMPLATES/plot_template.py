"""Template for plot module: *_plot.py

Instructions:
1. Replace MODEL_NAME with your model name (e.g., linear_regression_plot.py)
2. Update load_data() to load your specific dataset columns
3. Customize plot_model() for your visualization needs
4. Add type hints and docstrings
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load feature and target columns from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (feature_array, target_array) as numpy arrays
    """
    features = []
    targets = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # TODO: Update these column names
            features.append(float(row["FEATURE_COLUMN"]))
            targets.append(float(row["TARGET_COLUMN"]))

    return np.array(features), np.array(targets)


def plot_model(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Model Visualization",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot actual data points and model predictions.

    Args:
        x: Feature values (independent variable)
        y: Actual target values (dependent variable)
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

    # Prediction line
    plt.plot(x_sorted, y_pred_sorted, color="red", linewidth=2, label="Predictions")

    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
