"""Template for runner module: *_plot_runner.py

Instructions:
1. Replace MODEL_NAME with your model name (e.g., linear_regression_plot_runner.py)
2. Update csv_path with your data file location
3. Import your specific model class
4. Import your plot functions from the plot module
5. Customize the fitting and prediction logic
"""

from pathlib import Path
import sys

import numpy as np

# Add parent directories to path to enable src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# TODO: Update these imports
from src.machine_learning.my_model import MyModel
from src.visualizations.my_plot import load_data, plot_model


def main() -> None:
    """
    Load data, fit model, make predictions, and plot results.
    """
    # Define data path relative to script location
    # TODO: Update the data file name
    csv_path = Path(__file__).parent.parent.parent / "generated/data/raw/my_data.csv"

    # Load data
    x, y = load_data(csv_path)

    # TODO: Adjust reshape based on your model's input requirements
    # For single feature: reshape(-1, 1)
    # For multiple features: reshape(-1, n_features)
    x_reshaped = x.reshape(-1, 1)

    # Create and fit model
    model = MyModel()
    model.fit(x_reshaped.tolist(), y.tolist())

    # Make predictions
    y_pred = np.array(model.predict(x_reshaped.tolist()))

    # TODO: Print model parameters (optional)
    print("Model fitted successfully!")
    if hasattr(model, "intercept_"):
        print(f"Intercept: {model.intercept_:.4f}")
    if hasattr(model, "coef_"):
        print(f"Coefficients: {model.coef_}")

    # Plot results
    plot_model(x, y, y_pred, title="My Model Visualization")


if __name__ == "__main__":
    main()
