import pandas as pd

from visualizations.hypothesis_testing_plot import (
    plot_coefficients,
    plot_t_values,
    plot_one_minus_p,
)


def main() -> None:
    # Adjust path to your Advertising.csv
    df = pd.read_csv("../../generated/data/raw/Advertising-three-media.csv")

    # Assuming "Sales" is the response column
    response_col = "Sales"

    plot_coefficients(df, response_col=response_col)
    plot_t_values(df, response_col=response_col, n_bootstrap=500, random_state=0)
    plot_one_minus_p(df, response_col=response_col, n_bootstrap=500, random_state=0)


if __name__ == "__main__":
    main()
