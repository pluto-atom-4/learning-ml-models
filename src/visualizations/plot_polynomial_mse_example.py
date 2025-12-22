from src.visualizations.plot_polynomial_mse import plot_degree_vs_mse
from src.machine_learning.model_selection_polynomial_cv import find_best_degree_with_cv

train_errors, val_errors, cv_errors, best_val, best_cv = find_best_degree_with_cv(
    x_train, y_train, x_val, y_val, max_degree=10, k=5
)

plot_degree_vs_mse(train_errors, val_errors, cv_errors)
