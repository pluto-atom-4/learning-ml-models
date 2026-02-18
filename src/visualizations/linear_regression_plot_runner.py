from pathlib import Path
import sys

import numpy as np

# Add parent directories to path to enable src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.machine_learning.linear_regression import LinearRegression
from src.visualizations.linear_regression_plot import (
    load_advertising_data,
    plot_regression,
)


def main() -> None:
    """
    Load Advertising dataset, fit linear regression model, and plot results.
    """
    # Define data path relative to script location
    csv_path = Path(__file__).parent.parent.parent / "generated/data/raw/Advertising-three-media.csv"

    # Load data
    x, y = load_advertising_data(csv_path)

    # Reshape x for the model (needs 2D array for multivariate input)
    x_reshaped = x.reshape(-1, 1)

    # Create and fit model
    model = LinearRegression()
    model.fit(x_reshaped.tolist(), y.tolist())

    # Make predictions
    y_pred = np.array(model.predict(x_reshaped.tolist()))

    # Print model parameters
    print(f"Intercept (β₀): {model.intercept_:.4f}")
    print(f"Coefficient (β₁): {model.coef_[0]:.4f}")
    print(f"\nRegression equation: Sales = {model.intercept_:.4f} + {model.coef_[0]:.4f} × TV")

    # Plot results
    plot_regression(x, y, y_pred)


if __name__ == "__main__":
    main()
