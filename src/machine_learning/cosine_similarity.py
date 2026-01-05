from __future__ import annotations
import numpy as np


def cosine_similarity_one_by_one(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity manually using loops.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i] ** 2
        norm_b += b[i] ** 2

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero vector encountered")

    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def cosine_similarity_vectorized(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity using NumPy vectorization.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length")

    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)

    if norm == 0:
        raise ValueError("Zero vector encountered")

    return float(dot / norm)
