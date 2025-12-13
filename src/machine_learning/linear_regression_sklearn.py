from typing import List

from sklearn.linear_model import LinearRegression as SklearnLR


class LinearRegressionSklearn:
    """
    Wrapper around sklearn's LinearRegression for consistent API.
    """

    def __init__(self):
        self.model = SklearnLR()

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        self.model.fit(X, y)

    def predict(self, X: List[List[float]]) -> List[float]:
        return self.model.predict(X).tolist()
