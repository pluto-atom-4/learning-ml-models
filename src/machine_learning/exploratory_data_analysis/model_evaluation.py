import pandas as pd

try:
    # When executed as part of the package
    from .dataset_utils import get_absolute_path
except ImportError:  # pragma: no cover - fallback for direct execution
    # When executed directly: python model_evaluation.py
    from machine_learning.exploratory_data_analysis.dataset_utils import get_absolute_path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

def engineer_features(df):
    """Adds predictive features like symptom count to improve accuracy."""
    symptom_cols = ['cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']
    # Create a cumulative score of symptoms
    df['symptom_score'] = df[symptom_cols].sum(axis=1)
    return df

def load_and_prepare_datasets():
    """
    Load train/test datasets, apply feature engineering, and prepare feature/target splits.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) or (None, None, None, None) on error
    """
    try:
        train_df = pd.read_csv(get_absolute_path("covid_train.csv"))
        test_df = pd.read_csv(get_absolute_path("covid_test.csv"))
    except FileNotFoundError:
        print("Error: CSV files not found. Please run the data prep scripts first.")
        return None, None, None, None

    # 1. Apply Feature Engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # 2. Define Predictors and Response
    X_train = train_df.drop(columns=['Urgency'])
    y_train = train_df['Urgency']
    X_test = test_df.drop(columns=['Urgency'])
    y_test = test_df['Urgency']

    return X_train, y_train, X_test, y_test

def report_results(y_test, y_pred):
    """
    Output classification report and confusion matrix metrics (text-based only).

    Args:
        y_test: Actual target values
        y_pred: Predicted target values
    """
    # 1. Evaluation: Accuracy Score
    acc = accuracy_score(y_test, y_pred)
    print(f"--- Model Results ---")
    print(f"Overall Accuracy: {acc:.2%}\n")

    # 2. Evaluation: Classification Report
    # Shows Precision, Recall, and F1-Score for both classes
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No Urgency (0)', 'High Urgency (1)']))

    # 3. Evaluation: Confusion Matrix
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
