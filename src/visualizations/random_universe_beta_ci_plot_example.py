import pandas as pd
from visualizations.random_universe_beta_ci_plot import plot_beta_confidence_intervals
from  random_universe_beta_ci_plot import randomUniverse   # Replace with your actual import

def main():
    df = pd.read_csv("../../generated/data/raw/advertising.csv")

    plot_beta_confidence_intervals(
        df,
        RandomUniverse=randomUniverse,
        n_universes=300,
    )


if __name__ == "__main__":
    main()
