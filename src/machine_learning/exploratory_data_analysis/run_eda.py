from impute_data import load_and_impute
import plotting_utils as plots


def main():
    # 1. Get clean data
    df = load_and_impute("covid.csv")

    # 2. Execute plotting functions
    print("Generating EDA plots...")
    plots.plot_urgency_by_age(df)
    plots.plot_common_symptoms_urgent(df)
    plots.compare_cough_by_urgency(df)


if __name__ == "__main__":
    main()
