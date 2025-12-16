import numpy as np


def initialize_centroids(points, k, rng=None):
    """Randomly choose k points as initial centroids."""
    rng = rng or np.random.default_rng()
    indices = rng.choice(len(points), size=k, replace=False)
    return points[indices]


def assign_points(points, centroids):
    """Assign each point to the nearest centroid."""
    distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def compute_centroids(points, assignments, k):
    """Compute new centroids as the mean of assigned points."""
    return np.array([
        points[assignments == i].mean(axis=0)
        for i in range(k)
    ])


def kmeans(points, k, max_iters=100, rng=None):
    """Run K-means clustering in a pure functional style."""
    centroids = initialize_centroids(points, k, rng)
    assignments = None

    for _ in range(max_iters):
        assignments = assign_points(points, centroids)
        new_centroids = compute_centroids(points, assignments, k)

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids, assignments
