from __future__ import annotations
import time
import numpy as np
from src.machine_learning.cosine_similarity import (
    cosine_similarity_one_by_one,
    cosine_similarity_vectorized,
)


def benchmark():
    a = np.random.rand(1_000_000)
    b = np.random.rand(1_000_000)

    # One-by-one
    start = time.perf_counter()
    cosine_similarity_one_by_one(a, b)
    t1 = time.perf_counter() - start

    # Vectorized
    start = time.perf_counter()
    cosine_similarity_vectorized(a, b)
    t2 = time.perf_counter() - start

    print(f"One-by-one time: {t1:.6f} seconds")
    print(f"Vectorized time: {t2:.6f} seconds")
    print(f"Speedup: {t1 / t2:.2f}x")


if __name__ == "__main__":
    benchmark()
