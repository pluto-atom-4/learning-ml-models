import matplotlib.pyplot as plt
import numpy as np

def plot_degree_vs_mse(train_errors, val_errors, cv_errors=None, save_path=None):
    """
    Plot polynomial degree vs. MSE for training, validation, and optional CV.
    Pure functional: no mutation of inputs.
    """
    degrees = np.arange(len(train_errors))

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_errors, marker="o", label="Train MSE")
    plt.plot(degrees, val_errors, marker="o", label="Validation MSE")

    if cv_errors is not None:
        plt.plot(degrees, cv_errors, marker="o", label="Cross‑Validation MSE")

    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title("Degree vs. MSE")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
