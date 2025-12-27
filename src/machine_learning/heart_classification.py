import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_heart_dataset(csv_path):
    """
    Load Heart.csv and convert AHD to binary (Yes=1, No=0).
    """
    df = pd.read_csv(csv_path)
    df["AHD"] = df["AHD"].map({"Yes": 1, "No": 0})
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split into train/validation sets.
    Predictor: Age
    Response: AHD
    """
    X = df[["Age"]].values
    y = df["AHD"].values

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_knn(X_train, y_train, k=5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model


def fit_logistic(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Compute train and validation accuracy.
    """
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    return accuracy_score(y_train, train_pred), accuracy_score(y_val, val_pred)
