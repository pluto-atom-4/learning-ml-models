from pathlib import Path
from visualizations.poly_land_classifier_plot import plot_all_models


def main():
    csv_path = Path("../../generated/data/raw/land_data.csv")
    plot_all_models(csv_path)


if __name__ == "__main__":
    main()
