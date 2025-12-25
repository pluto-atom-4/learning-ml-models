import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from src.machine_learning.polynomial_regression import (
    fit_polynomial_regression,
    predict_polynomial_regression,
)


def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return df


def train_and_evaluate_models(csv_path, test_size=0.3, random_state=42):
    """
    Train 5 models and compute their MSE on the test set.
    Returns list of MSE values in model order.
    """
    df = load_dataset(csv_path)

    x = df["x"].values.reshape(-1, 1)
    y = df["y"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    errors = []

    # 1. Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_pred = lin_reg.predict(x_test)
    errors.append(mse(y_test, y_pred))

    # 2. Polynomial Regression degree 2
    w2 = fit_polynomial_regression(x_train.flatten(), y_train, degree=2)
    y_pred2 = predict_polynomial_regression(x_test.flatten(), w2)
    errors.append(mse(y_test, y_pred2))

    # 3. Polynomial Regression degree 5
    w5 = fit_polynomial_regression(x_train.flatten(), y_train, degree=5)
    y_pred5 = predict_polynomial_regression(x_test.flatten(), w5)
    errors.append(mse(y_test, y_pred5))

    # 4. kNN Regression (k=2)
    knn2 = KNeighborsRegressor(n_neighbors=2)
    knn2.fit(x_train, y_train)
    y_pred_knn2 = knn2.predict(x_test)
    errors.append(mse(y_test, y_pred_knn2))

    # 5. kNN Regression (k=20)
    knn20 = KNeighborsRegressor(n_neighbors=20)
    knn20.fit(x_train, y_train)
    y_pred_knn20 = knn20.predict(x_test)
    errors.append(mse(y_test, y_pred_knn20))

    return errors
