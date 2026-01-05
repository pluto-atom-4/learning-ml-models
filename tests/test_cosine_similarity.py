import numpy as np
import pytest
from src.machine_learning.cosine_similarity import (
    cosine_similarity_one_by_one,
    cosine_similarity_vectorized,
)


def test_basic_similarity():
    a = np.array([1, 0])
    b = np.array([1, 0])

    assert cosine_similarity_one_by_one(a, b) == pytest.approx(1.0)
    assert cosine_similarity_vectorized(a, b) == pytest.approx(1.0)


def test_orthogonal_vectors():
    a = np.array([1, 0])
    b = np.array([0, 1])

    assert cosine_similarity_one_by_one(a, b) == pytest.approx(0.0)
    assert cosine_similarity_vectorized(a, b) == pytest.approx(0.0)


def test_negative_similarity():
    a = np.array([1, 0])
    b = np.array([-1, 0])

    assert cosine_similarity_one_by_one(a, b) == pytest.approx(-1.0)
    assert cosine_similarity_vectorized(a, b) == pytest.approx(-1.0)


def test_vector_mismatch():
    a = np.array([1, 2])
    b = np.array([1])

    with pytest.raises(ValueError):
        cosine_similarity_one_by_one(a, b)

    with pytest.raises(ValueError):
        cosine_similarity_vectorized(a, b)


def test_zero_vector():
    a = np.array([0, 0])
    b = np.array([1, 2])

    with pytest.raises(ValueError):
        cosine_similarity_one_by_one(a, b)

    with pytest.raises(ValueError):
        cosine_similarity_vectorized(a, b)
