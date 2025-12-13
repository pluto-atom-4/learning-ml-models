"""
Silhouette Score Algorithm Explained Step-by-Step
--------------------------------------------------
The Silhouette Score is a machine learning metric that evaluates clustering quality by measuring
how well-separated clusters are. It quantifies whether points are closer to points in their own
cluster compared to points in other clusters. This metric ranges from -1 to +1, where higher values
indicate better-defined clusters. Silhouette Score is fundamental for validating clustering results,
hyperparameter tuning, and selecting the optimal number of clusters.

Here is how the process works:

1. **Cluster Organization**: Group all sample indices by their assigned cluster labels.
   - Create a mapping from cluster label to list of sample indices
   - Use defaultdict to efficiently store and access cluster memberships
   - This allows O(1) lookup for finding all points in a given cluster

2. **Intra-cluster Distance (a)**: Calculate average distance to points in the same cluster.
   - For each sample i, find all other points in its cluster
   - Compute Euclidean distance from sample i to each cluster member
   - a = mean of these distances (0 if sample is alone in cluster)
   - Lower a-values indicate tight, cohesive clusters

3. **Inter-cluster Distance (b)**: Find minimum average distance to other clusters.
   - For each sample i, calculate average distance to points in each other cluster
   - b = minimum of these average distances across all other clusters
   - This represents the nearest neighboring cluster
   - Higher b-values indicate clusters are well-separated

4. **Silhouette Coefficient**: Compute individual silhouette score for each sample.
   - s(i) = (b - a) / max(a, b) when max(a, b) > 0
   - Interpretation: s(i) = 1 means sample is far from other clusters
   - Interpretation: s(i) = 0 means sample is on cluster boundary
   - Interpretation: s(i) < 0 means sample may be assigned to wrong cluster
   - Special case: s(i) = 0 if sample is alone in its cluster

5. **Mean Silhouette Score**: Average silhouette coefficients across all samples.
   - Compute the mean of all individual silhouette scores
   - This represents overall clustering quality
   - Range: [-1, 1] where 1 is ideal, 0 is ambiguous, -1 is poor
   - Use this value to compare different clustering algorithms/parameters

6. **Interpretation and Use Cases**:
   - Score near 1: Clusters are very well-separated and cohesive
   - Score near 0: Clusters overlap or samples are on boundaries
   - Score near -1: Samples may be assigned to wrong clusters
   - Used for: Selecting optimal k in k-means, validating cluster quality

Example: 3 points in cluster A at (0,0), (1,0), (2,0); 2 points in cluster B at (10,0), (11,0)
- For point (0,0): a = avg(dist to (1,0), dist to (2,0)) ≈ 1.0
- For point (0,0): b = avg(dist to (10,0), dist to (11,0)) ≈ 10.5
- For point (0,0): s ≈ (10.5 - 1.0) / 10.5 ≈ 0.90
- Higher score indicates good clustering

Time Complexity: O(n² * d) where n = number of samples, d = number of features
  - Pairwise distance computation: O(n²) distances
  - Each distance calculation: O(d) for d-dimensional vectors
  - For each sample: compute distances to all other samples

Space Complexity: O(n) for cluster storage and scores list
  - Cluster mapping: O(n) to store all indices
  - Scores list: O(n) for individual silhouette scores
  - Distance calculations use temporary arrays

This algorithm is essential for clustering validation, model selection, and understanding
cluster quality in unsupervised machine learning tasks.
"""

from collections import defaultdict
from typing import List

import numpy as np


def silhouette_score(X: List[List[float]], labels: List[int]) -> float:
    """
    Compute the mean Silhouette Score for clustering results.

    Parameters:
    - X: List of feature vectors (n_samples x n_features)
    - labels: Cluster labels for each sample

    Returns:
    - Mean silhouette score across all samples
    """
    X = np.array(X)
    n_samples = len(X)
    if n_samples == 0:
        return 0.0

    # Group indices by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    scores = []
    for i in range(n_samples):
        label = labels[i]
        own_cluster = clusters[label]

        # a: average distance to points in the same cluster
        if len(own_cluster) > 1:
            a = np.mean([np.linalg.norm(X[i] - X[j]) for j in own_cluster if j != i])
        else:
            a = 0.0

        # b: minimum average distance to points in other clusters
        b = float("inf")
        for other_label, other_indices in clusters.items():
            if other_label == label:
                continue
            avg_dist = np.mean([np.linalg.norm(X[i] - X[j]) for j in other_indices])
            b = min(b, avg_dist)

        # Silhouette score for point i
        # If there are no other clusters, silhouette score is 0
        if b == float("inf"):
            s = 0.0
        elif max(a, b) > 0:
            s = (b - a) / max(a, b)
        else:
            s = 0.0
        scores.append(s)

    return float(np.mean(scores))
