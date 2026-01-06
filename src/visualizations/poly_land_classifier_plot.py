from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from src.machine_learning.multi_poly_land_classifier import (
    load_land_data,
    search_best_hyperparams,
)


def plot_decision_boundary(ax, model, poly, X, y, title: str):
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_poly = poly.transform(grid)
    Z = model.predict(grid_poly).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    ax.set_title(title)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")


def plot_all_models(csv_path: str):
    X, y = load_land_data(csv_path)

    degrees = [1, 2, 3, 4]
    C_values = [0.01, 0.1, 1, 10]

    results = search_best_hyperparams(X, y, degrees, C_values)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.ravel()

    for ax, (d, C, acc, model, poly) in zip(axes, results[:8]):
        title = f"deg={d}, C={C}, acc={acc:.2f}"
        plot_decision_boundary(ax, model, poly, X, y, title)

    plt.tight_layout()
    plt.show()
