import pytest

from machine_learning.silhouette_score import silhouette_score


def test_simple_clusters():
    # Two well-separated clusters
    X = [[0, 0], [0, 1], [10, 10], [10, 11]]
    labels = [0, 0, 1, 1]
    score = silhouette_score(X, labels)
    assert 0 <= score <= 1  # should be positive


def test_single_cluster():
    X = [[1, 2], [2, 3], [3, 4]]
    labels = [0, 0, 0]
    score = silhouette_score(X, labels)
    assert score == 0.0  # silhouette undefined for single cluster


def test_perfect_separation():
    X = [[0, 0], [0, 1], [100, 100], [101, 101]]
    labels = [0, 0, 1, 1]
    score = silhouette_score(X, labels)
    assert score > 0.9  # nearly perfect separation


def test_empty_input():
    assert silhouette_score([], []) == 0.0


def test_mixed_clusters():
    X = [[1, 1], [2, 2], [1.1, 1.2], [8, 8], [9, 9]]
    labels = [0, 0, 0, 1, 1]
    score = silhouette_score(X, labels)
    assert -1 <= score <= 1
