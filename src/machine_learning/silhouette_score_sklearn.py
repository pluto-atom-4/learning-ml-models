from typing import List

import numpy as np
from sklearn.metrics import silhouette_score as sklearn_silhouette_score


def silhouette_score_sklearn(X: List[List[float]], labels: List[int]) -> float:
    """
    Wrapper around sklearn.metrics.silhouette_score for consistency.
    """
    X = np.array(X)
    return float(sklearn_silhouette_score(X, labels))
