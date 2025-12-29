import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_smooth_decision_boundary(X_train, y_train, X_val, y_val, knn_model, log_model):
    """
    Smooth decision boundary visualization using linspace for continuous prediction curves.

    This visualization shows:
    - Training and validation data separately
    - Smooth decision boundaries for both models
    - Probability estimation curves for both models
    - All on a single 2x2 grid

    Parameters:
    -----------
    X_train : array-like
        Training feature data (Age)
    y_train : array-like
        Training labels (AHD: 0=No, 1=Yes)
    X_val : array-like
        Validation feature data (Age)
    y_val : array-like
        Validation labels (AHD: 0=No, 1=Yes)
    knn_model : sklearn model
        Trained kNN classifier with predict and predict_proba methods
    log_model : sklearn model
        Trained Logistic Regression model with predict and predict_proba methods
    """
    # Create dummy x for smooth curve plotting
    # Extend beyond min/max of observed values for better visualization
    x_min = np.min(np.vstack([X_train, X_val])) - 10
    x_max = np.max(np.vstack([X_train, X_val])) + 10
    x_dummy = np.linspace(x_min, x_max, 200).reshape(-1, 1)

    # Get predictions and probabilities on dummy data
    knn_dummy_pred = knn_model.predict(x_dummy)
    knn_dummy_proba = knn_model.predict_proba(x_dummy)[:, 1]

    log_dummy_pred = log_model.predict(x_dummy)
    log_dummy_proba = log_model.predict_proba(x_dummy)[:, 1]

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Heart Classification: Smooth Decision Boundaries & Probability Curves",
                 fontsize=18, fontweight="bold")

    # ===== Plot 1 (Top-Left): kNN Classifications =====
    ax = axes[0, 0]

    # Plot training data
    ax.scatter(X_train[y_train == 0], np.full(np.sum(y_train == 0), -0.15),
              color="lightblue", s=80, alpha=0.6, label="Train: No AHD", marker="o", edgecolors="darkblue")
    ax.scatter(X_train[y_train == 1], np.full(np.sum(y_train == 1), -0.15),
              color="lightcoral", s=80, alpha=0.6, label="Train: Yes AHD", marker="o", edgecolors="darkred")

    # Plot validation data
    ax.scatter(X_val[y_val == 0], np.full(np.sum(y_val == 0), 1.15),
              color="darkblue", s=100, alpha=0.7, label="Val: No AHD", marker="^", edgecolors="navy")
    ax.scatter(X_val[y_val == 1], np.full(np.sum(y_val == 1), 1.15),
              color="darkred", s=100, alpha=0.7, label="Val: Yes AHD", marker="^", edgecolors="maroon")

    # Plot smooth decision boundary
    ax.plot(x_dummy, knn_dummy_pred, color="blue", linewidth=3, label="k-NN Decision Boundary", alpha=0.8)
    ax.fill_between(x_dummy.flatten(), -0.3, knn_dummy_pred.flatten(), alpha=0.1, color="blue", label="Predicted: No AHD")
    ax.fill_between(x_dummy.flatten(), knn_dummy_pred.flatten(), 1.3, alpha=0.1, color="blue", label="Predicted: Yes AHD")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Boundary (0.5)")
    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Classification", fontsize=12, fontweight="bold")
    ax.set_title("(1) k-NN Classification Decision Boundary", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["No AHD", "Boundary", "Yes AHD"])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="x")

    # ===== Plot 2 (Top-Right): Logistic Regression Classifications =====
    ax = axes[0, 1]

    # Plot training data
    ax.scatter(X_train[y_train == 0], np.full(np.sum(y_train == 0), -0.15),
              color="lightblue", s=80, alpha=0.6, label="Train: No AHD", marker="o", edgecolors="darkblue")
    ax.scatter(X_train[y_train == 1], np.full(np.sum(y_train == 1), -0.15),
              color="lightcoral", s=80, alpha=0.6, label="Train: Yes AHD", marker="o", edgecolors="darkred")

    # Plot validation data
    ax.scatter(X_val[y_val == 0], np.full(np.sum(y_val == 0), 1.15),
              color="darkblue", s=100, alpha=0.7, label="Val: No AHD", marker="^", edgecolors="navy")
    ax.scatter(X_val[y_val == 1], np.full(np.sum(y_val == 1), 1.15),
              color="darkred", s=100, alpha=0.7, label="Val: Yes AHD", marker="^", edgecolors="maroon")

    # Plot smooth decision boundary
    ax.plot(x_dummy, log_dummy_pred, color="red", linewidth=3, label="Logistic Decision Boundary", alpha=0.8)
    ax.fill_between(x_dummy.flatten(), -0.3, log_dummy_pred.flatten(), alpha=0.1, color="red", label="Predicted: No AHD")
    ax.fill_between(x_dummy.flatten(), log_dummy_pred.flatten(), 1.3, alpha=0.1, color="red", label="Predicted: Yes AHD")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Boundary (0.5)")
    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Classification", fontsize=12, fontweight="bold")
    ax.set_title("(2) Logistic Regression Classification Decision Boundary", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["No AHD", "Boundary", "Yes AHD"])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="x")

    # ===== Plot 3 (Bottom-Left): kNN Probability Estimation =====
    ax = axes[1, 0]

    # Plot training data
    ax.scatter(X_train[y_train == 0], np.full(np.sum(y_train == 0), -0.05),
              color="lightblue", s=80, alpha=0.6, label="Train: No AHD", marker="o", edgecolors="darkblue")
    ax.scatter(X_train[y_train == 1], np.full(np.sum(y_train == 1), -0.05),
              color="lightcoral", s=80, alpha=0.6, label="Train: Yes AHD", marker="o", edgecolors="darkred")

    # Plot validation data
    ax.scatter(X_val[y_val == 0], np.full(np.sum(y_val == 0), 1.05),
              color="darkblue", s=100, alpha=0.7, label="Val: No AHD", marker="^", edgecolors="navy")
    ax.scatter(X_val[y_val == 1], np.full(np.sum(y_val == 1), 1.05),
              color="darkred", s=100, alpha=0.7, label="Val: Yes AHD", marker="^", edgecolors="maroon")

    # Plot probability curve
    ax.plot(x_dummy, knn_dummy_proba, color="blue", linewidth=3, label="k-NN Probability", alpha=0.8)
    ax.fill_between(x_dummy.flatten(), 0, knn_dummy_proba.flatten(), alpha=0.2, color="blue")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Threshold (0.5)")

    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability of AHD=Yes", fontsize=12, fontweight="bold")
    ax.set_title("(3) k-NN Probability Estimation Curve", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="x")

    # ===== Plot 4 (Bottom-Right): Logistic Probability Estimation =====
    ax = axes[1, 1]

    # Plot training data
    ax.scatter(X_train[y_train == 0], np.full(np.sum(y_train == 0), -0.05),
              color="lightblue", s=80, alpha=0.6, label="Train: No AHD", marker="o", edgecolors="darkblue")
    ax.scatter(X_train[y_train == 1], np.full(np.sum(y_train == 1), -0.05),
              color="lightcoral", s=80, alpha=0.6, label="Train: Yes AHD", marker="o", edgecolors="darkred")

    # Plot validation data
    ax.scatter(X_val[y_val == 0], np.full(np.sum(y_val == 0), 1.05),
              color="darkblue", s=100, alpha=0.7, label="Val: No AHD", marker="^", edgecolors="navy")
    ax.scatter(X_val[y_val == 1], np.full(np.sum(y_val == 1), 1.05),
              color="darkred", s=100, alpha=0.7, label="Val: Yes AHD", marker="^", edgecolors="maroon")

    # Plot probability curve
    ax.plot(x_dummy, log_dummy_proba, color="red", linewidth=3, label="Logistic Probability", alpha=0.8)
    ax.fill_between(x_dummy.flatten(), 0, log_dummy_proba.flatten(), alpha=0.2, color="red")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Threshold (0.5)")

    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability of AHD=Yes", fontsize=12, fontweight="bold")
    ax.set_title("(4) Logistic Regression Probability Estimation Curve", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.show()


def plot_overlapped_smooth_boundaries(X_train, y_train, X_val, y_val, knn_model, log_model):
    """
    Overlapped visualization of smooth decision boundaries and probability curves using linspace.

    This creates a comprehensive single-canvas visualization where:
    - Training and validation data shown as background scatter
    - Classification decision boundaries plotted as solid lines (left y-axis)
    - Probability estimation curves plotted as dashed lines (right y-axis)
    - Both models compared directly on the same canvas

    Parameters:
    -----------
    X_train : array-like
        Training feature data (Age)
    y_train : array-like
        Training labels (AHD: 0=No, 1=Yes)
    X_val : array-like
        Validation feature data (Age)
    y_val : array-like
        Validation labels (AHD: 0=No, 1=Yes)
    knn_model : sklearn model
        Trained kNN classifier with predict and predict_proba methods
    log_model : sklearn model
        Trained Logistic Regression model with predict and predict_proba methods
    """
    # Create dummy x for smooth curve plotting using linspace
    x_min = np.min(np.vstack([X_train, X_val])) - 10
    x_max = np.max(np.vstack([X_train, X_val])) + 10
    x_dummy = np.linspace(x_min, x_max, 200).reshape(-1, 1)

    # Get predictions and probabilities on dummy data
    knn_dummy_pred = knn_model.predict(x_dummy)
    knn_dummy_proba = knn_model.predict_proba(x_dummy)[:, 1]

    log_dummy_pred = log_model.predict(x_dummy)
    log_dummy_proba = log_model.predict_proba(x_dummy)[:, 1]

    # Create figure with primary and secondary y-axes
    fig, ax1 = plt.subplots(figsize=(16, 9))
    fig.suptitle("Heart Classification: Overlapped Smooth Boundaries & Probability Curves (Linspace)",
                 fontsize=18, fontweight="bold")

    # ===== Primary Y-Axis (Left): Classification Boundaries =====
    ax1.set_xlabel("Age", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Classification Decision Boundary", fontsize=12, fontweight="bold", color="black")
    ax1.tick_params(axis='y', labelcolor="black")

    # Plot training data as background dots
    ax1.scatter(X_train[y_train == 0], np.full(np.sum(y_train == 0), -0.2),
               color="lightblue", s=80, alpha=0.3, label="Train: No AHD", marker="o", edgecolors="darkblue", linewidth=0.3)
    ax1.scatter(X_train[y_train == 1], np.full(np.sum(y_train == 1), -0.2),
               color="lightcoral", s=80, alpha=0.3, label="Train: Yes AHD", marker="o", edgecolors="darkred", linewidth=0.3)

    # Plot validation data as background dots
    ax1.scatter(X_val[y_val == 0], np.full(np.sum(y_val == 0), 1.2),
               color="darkblue", s=100, alpha=0.4, label="Val: No AHD", marker="^", edgecolors="navy", linewidth=0.3)
    ax1.scatter(X_val[y_val == 1], np.full(np.sum(y_val == 1), 1.2),
               color="darkred", s=100, alpha=0.4, label="Val: Yes AHD", marker="^", edgecolors="maroon", linewidth=0.3)

    # Plot classification boundaries as smooth curves
    ax1.plot(x_dummy, knn_dummy_pred, color="blue", linewidth=3, label="k-NN (k=20) Classification",
            marker="", alpha=0.8, linestyle="-")
    ax1.plot(x_dummy, log_dummy_pred, color="red", linewidth=3, label="Logistic Classification",
            marker="", alpha=0.8, linestyle="-")

    # Add decision boundary reference
    ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=2, alpha=0.4, label="Decision Boundary (0.5)")
    ax1.set_ylim(-0.4, 1.4)
    ax1.grid(True, alpha=0.2, axis="x")

    # ===== Secondary Y-Axis (Right): Probability Curves =====
    ax2 = ax1.twinx()
    ax2.set_ylabel("Probability of AHD=Yes (Estimation)", fontsize=12, fontweight="bold", color="purple")
    ax2.tick_params(axis='y', labelcolor="purple")

    # Plot probability curves as dashed lines with semi-transparency
    ax2.plot(x_dummy, knn_dummy_proba, color="blue", linewidth=2.5, label="k-NN Probability",
            marker="", alpha=0.5, linestyle="--")
    ax2.plot(x_dummy, log_dummy_proba, color="red", linewidth=2.5, label="Logistic Probability",
            marker="", alpha=0.5, linestyle="--")

    # Fill between to show probability difference
    ax2.fill_between(x_dummy.flatten(), knn_dummy_proba, log_dummy_proba,
                     color="purple", alpha=0.08, label="Probability Divergence")

    # Add probability threshold reference
    ax2.axhline(y=0.5, color="purple", linestyle=":", linewidth=1.5, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    # ===== Combine legends from both axes =====
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left", ncol=2, framealpha=0.95)

    # ===== Add annotation guide =====
    ax1.text(0.02, 0.95, "Solid lines: Classifications (left axis) | Dashed lines: Probabilities (right axis) | Data: Bottom (train) & Top (val)",
            transform=ax1.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.4))

    plt.tight_layout()
    plt.show()


def plot_overlapped_classification_results(X_train, y_train, X_val, y_val, knn_pred, log_pred, knn_proba=None, log_proba=None):
    """
    Overlapped visualization with all plots on a single canvas using multiple y-axes.

    This creates a comprehensive multi-layer visualization where:
    - Validation data is shown as background scatter
    - Both model predictions are plotted as lines
    - Both model probabilities are plotted as semi-transparent lines
    - Multiple y-axes show data, predictions, and probabilities

    Parameters:
    -----------
    X_train : array-like
        Training feature data (Age)
    y_train : array-like
        Training labels (AHD: 0=No, 1=Yes)
    X_val : array-like
        Validation feature data (Age)
    y_val : array-like
        Validation labels (AHD: 0=No, 1=Yes)
    knn_pred : array-like
        kNN predictions (binary: 0 or 1)
    log_pred : array-like
        Logistic Regression predictions (binary: 0 or 1)
    knn_proba : array-like, optional
        kNN prediction probabilities (shape: n_samples x 2)
    log_proba : array-like, optional
        Logistic Regression prediction probabilities (shape: n_samples x 2)
    """
    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.suptitle("Heart Classification: Overlapped Multi-Layer Analysis", fontsize=18, fontweight="bold")

    # Sort validation data by Age
    sorted_indices_val = np.argsort(X_val.flatten())
    X_val_sorted = X_val[sorted_indices_val].flatten()
    y_val_sorted = y_val[sorted_indices_val]
    knn_pred_sorted = knn_pred[sorted_indices_val]
    log_pred_sorted = log_pred[sorted_indices_val]

    # ===== Primary Y-Axis (Left): Validation Data =====
    ax1.set_xlabel("Age", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Validation Data & Predictions", fontsize=12, fontweight="bold", color="black")
    ax1.tick_params(axis='y', labelcolor="black")

    # Plot validation data as background dots
    ax1.scatter(X_val[y_val == 0], np.full(np.sum(y_val == 0), -0.15),
               color="lightblue", s=100, alpha=0.4, label="Val: No AHD", marker="o", edgecolors="darkblue", linewidth=0.5)
    ax1.scatter(X_val[y_val == 1], np.full(np.sum(y_val == 1), -0.15),
               color="lightcoral", s=100, alpha=0.4, label="Val: Yes AHD", marker="o", edgecolors="darkred", linewidth=0.5)

    # Plot predictions on primary axis
    ax1.plot(X_val_sorted, knn_pred_sorted, color="blue", linewidth=3, label="k-NN (k=20) Predictions",
            marker="o", markersize=5, alpha=0.8, linestyle="-")
    ax1.plot(X_val_sorted, log_pred_sorted, color="red", linewidth=3, label="Logistic Predictions",
            marker="s", markersize=5, alpha=0.8, linestyle="-")

    ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=2, alpha=0.5, label="Decision Boundary (0.5)")
    ax1.set_ylim(-0.3, 1.3)
    ax1.grid(True, alpha=0.2, axis="x")

    # ===== Secondary Y-Axis (Right): Probabilities =====
    ax2 = ax1.twinx()
    ax2.set_ylabel("Probability of AHD=Yes", fontsize=12, fontweight="bold", color="purple")
    ax2.tick_params(axis='y', labelcolor="purple")

    if knn_proba is not None and log_proba is not None:
        knn_prob_sorted = knn_proba[sorted_indices_val, 1]
        log_prob_sorted = log_proba[sorted_indices_val, 1]

        # Plot probabilities with semi-transparent lines
        ax2.plot(X_val_sorted, knn_prob_sorted, color="blue", linewidth=2.5,
                label="k-NN Probability", marker="o", markersize=4, alpha=0.5, linestyle="--")
        ax2.plot(X_val_sorted, log_prob_sorted, color="red", linewidth=2.5,
                label="Logistic Probability", marker="s", markersize=4, alpha=0.5, linestyle="--")

        # Fill between probabilities to show difference
        ax2.fill_between(X_val_sorted, knn_prob_sorted, log_prob_sorted,
                         color="purple", alpha=0.1, label="Probability Difference")

        ax2.axhline(y=0.5, color="purple", linestyle=":", linewidth=1.5, alpha=0.4)
        ax2.set_ylim(-0.1, 1.1)

    # ===== Combine legends from both axes =====
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper left", ncol=2, framealpha=0.95)

    # ===== Add text annotations =====
    ax1.text(0.02, 0.98, "Data points at bottom | Predictions: Lines | Probabilities: Dashed",
            transform=ax1.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    plt.show()

    """
    Comprehensive visualization with all plots on a single canvas (3x2 grid).

    Parameters:
    -----------
    X_train : array-like
        Training feature data (Age)
    y_train : array-like
        Training labels (AHD: 0=No, 1=Yes)
    X_val : array-like
        Validation feature data (Age)
    y_val : array-like
        Validation labels (AHD: 0=No, 1=Yes)
    knn_pred : array-like
        kNN predictions (binary: 0 or 1)
    log_pred : array-like
        Logistic Regression predictions (binary: 0 or 1)
    knn_proba : array-like, optional
        kNN prediction probabilities (shape: n_samples x 2)
    log_proba : array-like, optional
        Logistic Regression prediction probabilities (shape: n_samples x 2)
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Heart Classification: Comprehensive Model Analysis", fontsize=18, fontweight="bold", y=0.995)

    # ===== Plot 1: Training and Validation Data Distribution =====
    ax = axes[0, 0]
    ax.scatter(X_train[y_train == 0], [0] * np.sum(y_train == 0), color="lightblue", label="Train: No AHD", alpha=0.6, s=60)
    ax.scatter(X_train[y_train == 1], [0] * np.sum(y_train == 1), color="lightcoral", label="Train: Yes AHD", alpha=0.6, s=60)
    ax.scatter(X_val[y_val == 0], [1] * np.sum(y_val == 0), color="darkblue", label="Val: No AHD", alpha=0.7, s=80, marker="^")
    ax.scatter(X_val[y_val == 1], [1] * np.sum(y_val == 1), color="darkred", label="Val: Yes AHD", alpha=0.7, s=80, marker="^")
    ax.set_xlabel("Age", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dataset Type", fontsize=11, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Training", "Validation"])
    ax.set_title("(1) Training & Validation Data Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")

    # ===== Plot 2: kNN Predictions (Line Chart) =====
    ax = axes[0, 1]
    sorted_indices_val = np.argsort(X_val.flatten())
    X_val_sorted = X_val[sorted_indices_val].flatten()
    knn_pred_sorted = knn_pred[sorted_indices_val]

    ax.plot(X_val_sorted, knn_pred_sorted, color="blue", linewidth=2.5, label="k-NN (k=20) Predictions", marker="o", markersize=5, alpha=0.7)
    ax.scatter(X_val, knn_pred, color="blue", s=60, alpha=0.5, edgecolors="darkblue", linewidth=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Boundary (0.5)")
    ax.set_xlabel("Age", fontsize=11, fontweight="bold")
    ax.set_ylabel("Prediction (0=No, 1=Yes)", fontsize=11, fontweight="bold")
    ax.set_title("(2) k-NN Predictions (Line Chart)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    # ===== Plot 3: Logistic Regression Predictions (Line Chart) =====
    ax = axes[1, 0]
    log_pred_sorted = log_pred[sorted_indices_val]

    ax.plot(X_val_sorted, log_pred_sorted, color="red", linewidth=2.5, label="Logistic Predictions", marker="s", markersize=5, alpha=0.7)
    ax.scatter(X_val, log_pred, color="red", s=60, alpha=0.5, edgecolors="darkred", linewidth=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Boundary (0.5)")
    ax.set_xlabel("Age", fontsize=11, fontweight="bold")
    ax.set_ylabel("Prediction (0=No, 1=Yes)", fontsize=11, fontweight="bold")
    ax.set_title("(3) Logistic Regression Predictions (Line Chart)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    # ===== Plot 4: kNN Probabilities (Line Chart) =====
    ax = axes[1, 1]
    if knn_proba is not None:
        knn_prob_sorted = knn_proba[sorted_indices_val, 1]

        ax.plot(X_val_sorted, knn_prob_sorted, color="blue", linewidth=2.5, label="k-NN Probability", marker="o", markersize=5, alpha=0.7)
        ax.scatter(X_val, knn_proba[:, 1], color="blue", s=60, alpha=0.5, edgecolors="darkblue", linewidth=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Boundary (0.5)")
        ax.fill_between(X_val_sorted, 0, knn_prob_sorted, color="blue", alpha=0.1)
        ax.set_xlabel("Age", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability of AHD=Yes", fontsize=11, fontweight="bold")
        ax.set_title("(4) k-NN Probabilities (Line Chart)", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
    else:
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # ===== Plot 5: Logistic Regression Probabilities (Line Chart) =====
    ax = axes[2, 0]
    if log_proba is not None:
        log_prob_sorted = log_proba[sorted_indices_val, 1]

        ax.plot(X_val_sorted, log_prob_sorted, color="red", linewidth=2.5, label="Logistic Probability", marker="s", markersize=5, alpha=0.7)
        ax.scatter(X_val, log_proba[:, 1], color="red", s=60, alpha=0.5, edgecolors="darkred", linewidth=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Boundary (0.5)")
        ax.fill_between(X_val_sorted, 0, log_prob_sorted, color="red", alpha=0.1)
        ax.set_xlabel("Age", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability of AHD=Yes", fontsize=11, fontweight="bold")
        ax.set_title("(5) Logistic Regression Probabilities (Line Chart)", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
    else:
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # ===== Plot 6: Combined Probability Comparison =====
    ax = axes[2, 1]
    if knn_proba is not None and log_proba is not None:
        knn_prob_sorted = knn_proba[sorted_indices_val, 1]
        log_prob_sorted = log_proba[sorted_indices_val, 1]

        ax.plot(X_val_sorted, knn_prob_sorted, color="blue", linewidth=2.5, label="k-NN (k=20)", marker="o", markersize=4, alpha=0.7)
        ax.plot(X_val_sorted, log_prob_sorted, color="red", linewidth=2.5, label="Logistic Regression", marker="s", markersize=4, alpha=0.7)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Boundary (0.5)")
        ax.fill_between(X_val_sorted, knn_prob_sorted, log_prob_sorted, color="purple", alpha=0.15, label="Probability Difference")
        ax.set_xlabel("Age", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability of AHD=Yes", fontsize=11, fontweight="bold")
        ax.set_title("(6) Probability Comparison (Combined)", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
    else:
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# Legacy functions for backwards compatibility
def plot_probability_line_chart(X_val, knn_proba, log_proba):
    """
    Line chart comparing kNN and Logistic Regression probabilities.

    Parameters:
    -----------
    X_val : array-like
        Validation feature data (Age)
    knn_proba : array-like
        kNN prediction probabilities (shape: n_samples x 2)
    log_proba : array-like
        Logistic Regression prediction probabilities (shape: n_samples x 2)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort by Age for better line visualization
    sorted_indices = np.argsort(X_val.flatten())
    X_sorted = X_val[sorted_indices].flatten()
    knn_prob_sorted = knn_proba[sorted_indices, 1]
    log_prob_sorted = log_proba[sorted_indices, 1]

    # Plot lines for both models
    ax.plot(X_sorted, knn_prob_sorted, color="blue", linewidth=2.5, label="k-NN (k=20)", marker="o", markersize=4, alpha=0.7)
    ax.plot(X_sorted, log_prob_sorted, color="red", linewidth=2.5, label="Logistic Regression", marker="s", markersize=4, alpha=0.7)

    # Add decision boundary
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="Decision Boundary (0.5)")

    # Styling
    ax.set_xlabel("Age", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability of AHD=Yes", fontsize=12, fontweight="bold")
    ax.set_title("kNN vs Logistic Regression: Probability Comparison (Line Chart)", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    plt.show()


def plot_classification_results(X, y, X_val, knn_pred, log_pred, knn_proba=None, log_proba=None):
    """
    Scatterplot of Age vs AHD predictions and probabilities for both models.

    Parameters:
    -----------
    X : array-like
        Training feature data (typically Age)
    y : array-like
        Training labels (AHD: 0=No, 1=Yes)
    X_val : array-like
        Validation feature data
    knn_pred : array-like
        kNN predictions (binary: 0 or 1)
    log_pred : array-like
        Logistic Regression predictions (binary: 0 or 1)
    knn_proba : array-like, optional
        kNN prediction probabilities (shape: n_samples x 2)
    log_proba : array-like, optional
        Logistic Regression prediction probabilities (shape: n_samples x 2)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("k-NN vs Logistic Regression: Predictions & Probabilities", fontsize=16, fontweight="bold")

    # ===== Plot 1: kNN Predictions =====
    ax = axes[0, 0]
    ax.scatter(X, y, color="black", label="Training Data", alpha=0.5, s=50)
    ax.scatter(X_val, knn_pred, color="blue", label="kNN Predictions", alpha=0.7, s=80, marker="s")
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("AHD Prediction (0=No, 1=Yes)", fontsize=11)
    ax.set_title("k-NN Predictions (k=20)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)

    # ===== Plot 2: Logistic Regression Predictions =====
    ax = axes[0, 1]
    ax.scatter(X, y, color="black", label="Training Data", alpha=0.5, s=50)
    ax.scatter(X_val, log_pred, color="red", label="Logistic Predictions", alpha=0.7, s=80, marker="^")
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("AHD Prediction (0=No, 1=Yes)", fontsize=11)
    ax.set_title("Logistic Regression Predictions", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)

    # ===== Plot 3: kNN Probability (Class 1) =====
    ax = axes[1, 0]
    if knn_proba is not None:
        scatter = ax.scatter(X_val, knn_proba[:, 1], c=knn_proba[:, 1], cmap="Blues",
                            s=80, alpha=0.7, edgecolors="darkblue", linewidth=0.5, label="Probability of AHD=Yes")
        ax.set_xlabel("Age", fontsize=11)
        ax.set_ylabel("Probability of AHD=Yes", fontsize=11)
        ax.set_title("k-NN Probability (Class 1)", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Decision Boundary (0.5)")
        ax.legend(fontsize=10)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Probability", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No probability data available", ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    # ===== Plot 4: Logistic Regression Probability (Class 1) =====
    ax = axes[1, 1]
    if log_proba is not None:
        scatter = ax.scatter(X_val, log_proba[:, 1], c=log_proba[:, 1], cmap="Reds",
                            s=80, alpha=0.7, edgecolors="darkred", linewidth=0.5, label="Probability of AHD=Yes")
        ax.set_xlabel("Age", fontsize=11)
        ax.set_ylabel("Probability of AHD=Yes", fontsize=11)
        ax.set_title("Logistic Regression Probability (Class 1)", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Decision Boundary (0.5)")
        ax.legend(fontsize=10)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Probability", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No probability data available", ha="center", va="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
