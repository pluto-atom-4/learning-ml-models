"""
Example script demonstrating the bias-variance tradeoff with polynomial regression.

This script:
  1. Loads the noisy population dataset
  2. Samples a subset of the data
  3. Fits polynomial models of different degrees
  4. Visualizes the predictions and bias-variance tradeoff
"""

import os
from src.machine_learning.bias_variance_polynomial import (
    load_noisy_population,
    sample_dataset,
    compute_polynomial_predictions,
)
from src.visualizations.plot_bias_variance import plot_bias_variance


def main():
    """
    Main function to run the bias-variance visualization example.
    """
    # Define the CSV file path relative to this script
    # From: src/visualizations/plot_bias_variance_example.py
    # To: generated/data/raw/noisypopulation.csv
    csv_path = "../../generated/data/raw/noisypopulation.csv"

    # Verify the file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Absolute path would be: {os.path.abspath(csv_path)}")
        return

    # Load the dataset
    print(f"Loading data from: {csv_path}")
    f_true, x, y = load_noisy_population(csv_path)

    # Sample a subset of the data for fitting
    sample_size = 50
    random_state = 42
    x_sample, y_sample = sample_dataset(x, y, sample_size=sample_size, random_state=random_state)

    print(f"Loaded {len(x)} samples, using {sample_size} for training")

    # Define polynomial degrees to compare
    degrees = [1, 2, 3, 5, 10]

    # Compute polynomial predictions for each degree
    print(f"Computing predictions for polynomial degrees: {degrees}")
    predictions, x_pred = compute_polynomial_predictions(
        x_sample, y_sample, degrees=degrees, num_points=100
    )

    # Interpolate the true function at prediction points
    # (using the 'f' column from the original dataset)
    from scipy.interpolate import interp1d

    f_interp = interp1d(x, f_true, kind="cubic", fill_value="extrapolate")
    true_f = f_interp(x_pred)

    # Generate the plot
    print("Generating plot...")
    save_path = "../../generated/images/bias_variance_example.png"

    # Ensure the output directory exists
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)

    plot_bias_variance(
        x_sample=x_sample,
        y_sample=y_sample,
        x_pred=x_pred,
        predictions=predictions,
        true_f=true_f,
        save_path=save_path,
    )

    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()

