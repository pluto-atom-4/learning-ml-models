from src.visualizations.plot_polynomial_models import (
    load_dataset,
    plot_polynomial_models,
)

csv_path = "../../generated/data/raw/dataset.csv"

x, y = load_dataset(csv_path)

plot_polynomial_models(x, y, max_degree=5)
