from pathlib import Path

from sklearn.model_selection import train_test_split
from impute_data import load_and_impute
from dataset_utils import get_absolute_path


def export_train_test(df):
    # Split: 70% train, 30% test, random_state=60
    train_df, test_df = train_test_split(df, test_size=0.30, random_state=60)

    # Save to CSV without default indices
    train_df.to_csv(get_absolute_path("covid_train.csv"), index=False)
    test_df.to_csv(get_absolute_path("covid_test.csv"), index=False)

    print("Files saved: covid_train.csv and covid_test.csv")


if __name__ == "__main__":
    clean_df = load_and_impute("covid.csv")
    export_train_test(clean_df)
