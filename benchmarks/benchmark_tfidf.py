import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.machine_learning.tfidf import compute_tfidf


def benchmark_custom(corpus, repeat=5):
    """Benchmark your pure functional TF-IDF implementation."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = compute_tfidf(corpus)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)


def benchmark_sklearn(corpus, repeat=5):
    """Benchmark scikit-learn's TfidfVectorizer."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        vectorizer = TfidfVectorizer()
        _ = vectorizer.fit_transform(corpus)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)


def main():
    # Create a synthetic corpus
    n_docs = 5000
    corpus = [
        "machine learning models use tf idf for text processing"
        for _ in range(n_docs)
    ]

    custom_time = benchmark_custom(corpus)
    sklearn_time = benchmark_sklearn(corpus)

    print("\nTF-IDF Benchmark")
    print("---------------------------")
    print(f"Documents: {n_docs}")
    print()
    print(f"Custom implementation: {custom_time:.6f} seconds")
    print(f"Scikit-learn version:  {sklearn_time:.6f} seconds")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
