import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from dataset_utils import get_absolute_path


def engineer_features(df):
    """Adds predictive features like symptom count to improve accuracy."""
    symptom_cols = ['cough', 'fever', 'chills', 'sore_throat', 'headache', 'fatigue']
    # Create a cumulative score of symptoms
    df['symptom_score'] = df[symptom_cols].sum(axis=1)
    return df

def run_prediction_pipeline():
    # 1. Load Datasets
    try:
        train_df = pd.read_csv(get_absolute_path("covid_train.csv"))
        test_df = pd.read_csv(get_absolute_path("covid_test.csv"))
    except FileNotFoundError:
        print("Error: CSV files not found. Please run the data prep scripts first.")
        return

    # 2. Apply Feature Engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # 3. Define Predictors and Response
    X_train = train_df.drop(columns=['Urgency'])
    y_train = train_df['Urgency']
    X_test = test_df.drop(columns=['Urgency'])
    y_test = test_df['Urgency']

    # 4. Initialize and Fit kNN (using k=5 as a baseline)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # 5. Predict
    y_pred = knn.predict(X_test)

    # 6. Evaluation: Accuracy Score
    acc = accuracy_score(y_test, y_pred)
    print(f"--- Model Results ---")
    print(f"Overall Accuracy: {acc:.2%}\n")

    # 7. Evaluation: Classification Report
    # Shows Precision, Recall, and F1-Score for both classes
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No Urgency (0)', 'High Urgency (1)']))

    # 8. Evaluation: Confusion Matrix
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualizing the Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Urgency', 'High Urgency'])
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Confusion Matrix: COVID-19 Hospitalization Urgency")
    plt.show()

if __name__ == "__main__":
    run_prediction_pipeline()
