from typing import List

import numpy as np
from sklearn.neighbors import KNeighborsRegressor as SklearnKNN


class KNeighborsRegressorSklearn:
    """
    Wrapper around sklearn.neighbors.KNeighborsRegressor for consistency.
    """

    def __init__(self, n_neighbors: int = 5):
        self.model = SklearnKNN(n_neighbors=n_neighbors)

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        self.model.fit(X, y)

    def predict(self, X: List[List[float]]) -> List[float]:
        return self.model.predict(X).tolist()
