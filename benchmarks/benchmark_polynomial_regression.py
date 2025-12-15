import time
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from src.machine_learning.polynomial_regression import (
    fit_polynomial_regression,
    predict_polynomial_regression,
)


def benchmark_custom(x, y, degree, repeat=5):
    """Benchmark your pure functional implementation."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        w = fit_polynomial_regression(x, y, degree)
        _ = predict_polynomial_regression(x, w)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)


def benchmark_sklearn(x, y, degree, repeat=5):
    """Benchmark scikit-learn's PolynomialFeatures + LinearRegression."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression()
        model.fit(X_poly, y)
        _ = model.predict(X_poly)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)


def main():
    # Synthetic dataset
    n_samples = 50_000
    degree = 5

    x = np.random.rand(n_samples)
    y = 2 + 3 * x + 0.5 * x**2 + np.random.normal(0, 0.1, size=n_samples)

    custom_time = benchmark_custom(x, y, degree)
    sklearn_time = benchmark_sklearn(x, y, degree)

    print("\nPolynomial Regression Benchmark")
    print("--------------------------------")
    print(f"Samples: {n_samples}")
    print(f"Degree:  {degree}")
    print()
    print(f"Custom implementation: {custom_time:.6f} seconds")
    print(f"Scikit-learn version:  {sklearn_time:.6f} seconds")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
