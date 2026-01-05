from pathlib import Path
from src.visualizations.vector_angle_plot import load_and_plot


def main() -> None:
    csv_path = Path("../../generated/data/raw/vector_examples.csv")
    load_and_plot(csv_path)


if __name__ == "__main__":
    main()
