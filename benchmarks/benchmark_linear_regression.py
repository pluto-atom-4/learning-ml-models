import time

import numpy as np
from sklearn.metrics import mean_squared_error as sklearn_mse

from machine_learning.linear_regression import LinearRegression
from machine_learning.linear_regression_sklearn import LinearRegressionSklearn
from machine_learning.mse import mean_squared_error


def generate_data(n_samples=5000, n_features=10):
    """
    Generate synthetic linear regression data.
    """
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    return X.tolist(), y.tolist()


def benchmark(model_class, X_train, y_train, X_test, y_test, trials=3):
    durations = []
    mses = []

    for _ in range(trials):
        model = model_class()
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        end = time.time()

        durations.append(end - start)
        mses.append(mean_squared_error(y_test, preds))

    return {
        "avg_time": sum(durations) / trials,
        "min_time": min(durations),
        "max_time": max(durations),
        "avg_mse": sum(mses) / trials,
    }


def benchmark_sklearn(X_train, y_train, X_test, y_test, trials=3):
    durations = []
    mses = []

    for _ in range(trials):
        model = LinearRegressionSklearn()
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        end = time.time()

        durations.append(end - start)
        mses.append(sklearn_mse(y_test, preds))

    return {
        "avg_time": sum(durations) / trials,
        "min_time": min(durations),
        "max_time": max(durations),
        "avg_mse": sum(mses) / trials,
    }


def run_benchmark():
    X_train, y_train = generate_data(n_samples=4000, n_features=20)
    X_test, y_test = generate_data(n_samples=1000, n_features=20)

    print("\nBenchmarking Linear Regression Implementations...\n")

    custom_stats = benchmark(LinearRegression, X_train, y_train, X_test, y_test)
    sklearn_stats = benchmark_sklearn(X_train, y_train, X_test, y_test)

    print("⏱️ Timing Results")
    print(f"{'Model':<20} {'Avg Time':<12} {'Min':<12} {'Max':<12} {'Avg MSE'}")
    print(f"{'Custom':<20} {custom_stats['avg_time']:.5f}   {custom_stats['min_time']:.5f}   {custom_stats['max_time']:.5f}   {custom_stats['avg_mse']:.5f}")
    print(f"{'Scikit-Learn':<20} {sklearn_stats['avg_time']:.5f}   {sklearn_stats['min_time']:.5f}   {sklearn_stats['max_time']:.5f}   {sklearn_stats['avg_mse']:.5f}")


if __name__ == "__main__":
    run_benchmark()
