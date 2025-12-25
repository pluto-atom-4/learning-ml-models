import matplotlib.pyplot as plt
import numpy as np


def plot_regularization_coefficients(coefficients, feature_names, save_path=None):
    """
    coefficients: dict with keys ["linear", "lasso", "ridge"]
    feature_names: list of predictor names
    """
    models = list(coefficients.keys())
    num_features = len(feature_names)

    x_positions = np.arange(num_features)

    plt.figure(figsize=(10, 6))

    for model in models:
        plt.plot(
            x_positions,
            coefficients[model],
            marker="o",
            label=f"{model.capitalize()} Coefficients"
        )

    plt.xticks(x_positions, feature_names, rotation=45)
    plt.ylabel("Coefficient Value")
    plt.title("Linear vs Lasso vs Ridge Coefficients")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
