from machine_learning.regularization_comparison import compare_regularization_models
from visualizations.plot_regularization_coefficients import plot_regularization_coefficients

csv_path = "generated/data/raw/dataset.csv"

coeffs, errors, feature_names = compare_regularization_models(csv_path, alpha=1.0)

print("Validation MSE:", errors)

plot_regularization_coefficients(coeffs, feature_names)
