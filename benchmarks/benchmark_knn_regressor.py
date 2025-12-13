import time

import numpy as np

from machine_learning.knn_regressor import KNeighborsRegressor
from machine_learning.knn_regressor_sklearn import KNeighborsRegressorSklearn


def generate_data(n_samples=1000, n_features=5):
    """
    Generate synthetic regression dataset.
    """
    X = np.random.rand(n_samples, n_features) * 10
    y = np.sum(X, axis=1) + np.random.randn(n_samples)  # linear-ish target with noise
    return X.tolist(), y.tolist()

def benchmark(model_class, X_train, y_train, X_test, trials=3):
    durations = []
    for _ in range(trials):
        model = model_class(n_neighbors=5)
        start = time.time()
        model.fit(X_train, y_train)
        _ = model.predict(X_test)
        end = time.time()
        durations.append(end - start)
    return {
        "average": sum(durations) / trials,
        "min": min(durations),
        "max": max(durations)
    }

def run_benchmark():
    X_train, y_train = generate_data(n_samples=2000, n_features=10)
    X_test, _ = generate_data(n_samples=200, n_features=10)

    print(f"Benchmarking KNeighborsRegressor with {len(X_train)} training samples...\n")

    custom_stats = benchmark(KNeighborsRegressor, X_train, y_train, X_test)
    sklearn_stats = benchmark(KNeighborsRegressorSklearn, X_train, y_train, X_test)

    print("⏱️ Timing Results:")
    print(f"{'Implementation':<25} {'Avg':<10} {'Min':<10} {'Max':<10}")
    print(f"{'Custom':<25} {custom_stats['average']:.5f}  {custom_stats['min']:.5f}  {custom_stats['max']:.5f}")
    print(f"{'Sklearn':<25} {sklearn_stats['average']:.5f}  {sklearn_stats['min']:.5f}  {sklearn_stats['max']:.5f}")

if __name__ == "__main__":
    run_benchmark()
