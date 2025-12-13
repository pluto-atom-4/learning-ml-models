import time

import numpy as np

from machine_learning.silhouette_score import silhouette_score
from machine_learning.silhouette_score_sklearn import silhouette_score_sklearn


def generate_data(n_samples=500, n_features=5, n_clusters=3):
    """
    Generate synthetic clustered data.
    """
    X = []
    labels = []
    for cluster_id in range(n_clusters):
        center = np.random.rand(n_features) * 10
        for _ in range(n_samples // n_clusters):
            point = center + np.random.randn(n_features)
            X.append(point.tolist())
            labels.append(cluster_id)
    return X, labels

def benchmark(func, X, labels, trials=3):
    durations = []
    for _ in range(trials):
        start = time.time()
        _ = func(X, labels)
        end = time.time()
        durations.append(end - start)
    return {
        "average": sum(durations) / trials,
        "min": min(durations),
        "max": max(durations)
    }

def run_benchmark():
    X, labels = generate_data(n_samples=600, n_features=10, n_clusters=3)
    print(f"Benchmarking Silhouette Score with {len(X)} samples, {len(X[0])} features...\n")

    custom_stats = benchmark(silhouette_score, X, labels)
    sklearn_stats = benchmark(silhouette_score_sklearn, X, labels)

    print("⏱️ Timing Results:")
    print(f"{'Implementation':<25} {'Avg':<10} {'Min':<10} {'Max':<10}")
    print(f"{'Custom':<25} {custom_stats['average']:.5f}  {custom_stats['min']:.5f}  {custom_stats['max']:.5f}")
    print(f"{'Sklearn':<25} {sklearn_stats['average']:.5f}  {sklearn_stats['min']:.5f}  {sklearn_stats['max']:.5f}")

if __name__ == "__main__":
    run_benchmark()
