import matplotlib.pyplot as plt
import numpy as np


def plot_classification_results(X, y, X_val, knn_pred, log_pred):
    """
    Scatterplot of Age vs AHD predictions.
    """
    plt.figure(figsize=(10, 6))

    # True labels
    plt.scatter(X, y, color="black", label="True Labels", alpha=0.6)

    # kNN predictions
    plt.scatter(X_val, knn_pred, color="blue", label="kNN Predictions", alpha=0.7)

    # Logistic predictions
    plt.scatter(X_val, log_pred, color="red", label="Logistic Predictions", alpha=0.7)

    plt.xlabel("Age")
    plt.ylabel("AHD (0=No, 1=Yes)")
    plt.title("k-NN vs Logistic Regression Predictions")
    plt.legend()
    plt.grid(True)
    plt.show()
