from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Compute angle in degrees between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        raise ValueError("Zero vector encountered")
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def plot_vector_angle(a: np.ndarray, b: np.ndarray, title: str = "") -> None:
    """Plot two vectors and the angle between them."""
    angle = compute_angle(a, b)

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Plot vectors
    ax.arrow(0, 0, a[0], a[1], head_width=0.1, color="blue", length_includes_head=True)
    ax.arrow(0, 0, b[0], b[1], head_width=0.1, color="red", length_includes_head=True)

    # Annotate
    plt.text(0.1, 0.1, f"Angle = {angle:.2f}°", fontsize=12)

    # Axes settings
    max_val = max(np.linalg.norm(a), np.linalg.norm(b)) + 1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect("equal", adjustable="box")

    plt.title(title or "Angle Between Vectors")
    plt.grid(True)
    plt.show()


def load_and_plot(csv_path: str | Path) -> None:
    """Load dataset and plot each vector pair."""
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        a = np.array([row["x1"], row["x2"]])
        b = np.array([row["y1"], row["y2"]])

        title = f"Vector Pair {int(row['vector_id'])}"
        plot_vector_angle(a, b, title=title)
