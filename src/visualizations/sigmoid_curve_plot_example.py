from pathlib import Path
from src.visualizations.sigmoid_curve_plot import plot_sigmoid


def main() -> None:
    # Example: simulate a few different sigmoid shapes
    plot_sigmoid(x_range=(-10, 10), L=1.0, k=1.0, x0=0.0,
                 title="Standard Sigmoid Curve")

    plot_sigmoid(x_range=(-10, 10), L=1.0, k=2.0, x0=0.0,
                 title="Steeper Sigmoid (k=2)")

    plot_sigmoid(x_range=(-10, 10), L=1.0, k=1.0, x0=2.0,
                 title="Shifted Sigmoid (x0=2)")


if __name__ == "__main__":
    main()
