import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def compare_regularization_models(csv_path, alpha=1.0, test_size=0.3, random_state=42):
    """
    Returns:
        coefficients: dict with keys ["linear", "lasso", "ridge"]
        errors: dict with keys ["linear", "lasso", "ridge"]
        feature_names: list of predictor names
    """
    df = load_dataset(csv_path)

    X = df.drop(columns=["y"]).values
    y = df["y"].values
    feature_names = list(df.drop(columns=["y"]).columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    coefficients = {}
    errors = {}

    # Linear Regression
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred = lin.predict(X_val)
    coefficients["linear"] = lin.coef_
    errors["linear"] = mse(y_val, y_pred)

    # Lasso
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_val)
    coefficients["lasso"] = lasso.coef_
    errors["lasso"] = mse(y_val, y_pred_lasso)

    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_val)
    coefficients["ridge"] = ridge.coef_
    errors["ridge"] = mse(y_val, y_pred_ridge)

    return coefficients, errors, feature_names
