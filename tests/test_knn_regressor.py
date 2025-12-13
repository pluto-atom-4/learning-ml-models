import numpy as np
import pytest

from machine_learning.knn_regressor import KNeighborsRegressor


def test_knn_regressor_basic():
    X_train = [[1],[2],[3],[4],[5]]
    y_train = [1,2,3,4,5]
    model = KNeighborsRegressor(n_neighbors=2)
    model.fit(X_train, y_train)
    preds = model.predict([[1.5],[3.5]])
    # Predictions should be averages of nearest neighbors
    assert np.isclose(preds[0], 1.5)  # neighbors [1,2]
    assert np.isclose(preds[1], 3.5)  # neighbors [3,4]

def test_knn_regressor_single_neighbor():
    X_train = [[0],[10]]
    y_train = [0,10]
    model = KNeighborsRegressor(n_neighbors=1)
    model.fit(X_train, y_train)
    preds = model.predict([[1],[9]])
    assert preds == [0,10]

def test_knn_regressor_invalid_neighbors():
    with pytest.raises(ValueError):
        KNeighborsRegressor(n_neighbors=0)

def test_knn_regressor_not_fitted():
    model = KNeighborsRegressor(n_neighbors=2)
    with pytest.raises(ValueError):
        model.predict([[1]])
