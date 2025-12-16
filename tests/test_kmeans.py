import numpy as np
from src.machine_learning.kmeans import (
    initialize_centroids,
    assign_points,
    compute_centroids,
    kmeans,
)


def test_initialize_centroids():
    points = np.array([[1, 2], [3, 4], [5, 6]])
    centroids = initialize_centroids(points, k=2, rng=np.random.default_rng(0))
    assert centroids.shape == (2, 2)
    # Check that all centroids are from the original points
    assert all(any((centroids[i] == points).all(axis=1)) for i in range(len(centroids)))


def test_assign_points():
    points = np.array([[0, 0], [10, 10]])
    centroids = np.array([[0, 0], [10, 10]])
    assignments = assign_points(points, centroids)
    assert np.array_equal(assignments, np.array([0, 1]))


def test_compute_centroids():
    points = np.array([[0, 0], [2, 2], [10, 10], [12, 12]])
    assignments = np.array([0, 0, 1, 1])
    centroids = compute_centroids(points, assignments, k=2)
    assert np.allclose(centroids[0], np.array([1, 1]))
    assert np.allclose(centroids[1], np.array([11, 11]))


def test_kmeans_converges():
    points = np.array([
        [0, 0], [1, 1], [2, 2],
        [10, 10], [11, 11], [12, 12]
    ])
    centroids, assignments = kmeans(points, k=2, rng=np.random.default_rng(0))

    # Expect two clusters
    assert len(set(assignments)) == 2

    # Centroids should be near the cluster means
    assert any(np.allclose(c, np.array([1, 1]), atol=1e-1) for c in centroids)
    assert any(np.allclose(c, np.array([11, 11]), atol=1e-1) for c in centroids)
