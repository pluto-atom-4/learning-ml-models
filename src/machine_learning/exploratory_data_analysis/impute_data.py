import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer


def load_and_impute(file_path="covid.csv"):
    # Load dataset
    # Get absolute path to the sample resume file
    root_dir = Path(__file__).parent.parent.parent.parent
    #root_dir = script_dir.parent.parent.parent
    abs_file_path = root_dir / "generated" / "data" / "raw" / file_path
    print(abs_file_path)
    df = pd.read_csv(abs_file_path)

    # Check for missing values
    missing_rows = df.isnull().any(axis=1).sum()
    print(f"Number of rows with missing values: {missing_rows}")

    # Perform kNN Imputation (k=5)
    # Urgency is the response variable, but included in imputation context
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(df)

    # Return clean dataframe with original column names
    return pd.DataFrame(imputed_data, columns=df.columns)


if __name__ == "__main__":
    clean_df = load_and_impute()
    clean_df.head()
    print("Imputation complete.")
