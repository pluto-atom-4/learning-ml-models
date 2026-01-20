from pathlib import Path
import pandas as pd
from src.visualizations.random_universe_beta_plot import (
    plot_beta_distributions,
    plot_original_vs_parallel,
RandomUniverse
)

def main():
    df = pd.read_csv("../../generated/data/raw/advertising.csv")

    plot_beta_distributions(df, RandomUniverse, n_universes=300)
    plot_original_vs_parallel(df, RandomUniverse, n_samples=5)


if __name__ == "__main__":
    main()
