import numpy as np
from machine_learning.custom_random_survival_forest import (
    load_waltons_dataset,
    SurvivalTree,
    RandomSurvivalForest,
    km_estimator,
)


def test_km_estimator_basic():
    # Simple synthetic data
    times = np.array([1, 2, 3, 4])
    events = np.array([1, 1, 0, 1])

    km_t, km_s = km_estimator(times, events)

    # KM curve must be non-increasing
    assert np.all(np.diff(km_s) <= 1e-12)

    # Survival values must be between 0 and 1
    assert np.all((km_s >= 0) & (km_s <= 1))


def test_survival_tree_fit_predict():
    X = np.array([[0], [1], [0], [1]])
    T = np.array([5, 6, 7, 8])
    E = np.array([1, 1, 0, 1])

    tree = SurvivalTree(max_depth=3, min_samples_split=1, max_features=1)
    tree.fit(X, T, E)

    # Predict for a sample
    km_t, km_s = tree.predict_survival(np.array([0]))

    assert len(km_t) == len(km_s)
    assert np.all((km_s >= 0) & (km_s <= 1))


def test_rsf_fit_predict():
    X, T, E, _ = load_waltons_dataset()

    model = RandomSurvivalForest(
        n_estimators=5,
        max_depth=3,
        min_samples_split=5,
        max_features=1,
    )
    model.fit(X, T, E)

    # Predict survival for a single sample
    # Sample must have same number of features as X (2 features for one-hot encoded group)
    x_sample = np.array([1.0, 0.0])
    t, s = model.predict_survival_function(x_sample)

    # Survival curve must be valid
    assert len(t) == len(s)
    assert np.all((s >= 0) & (s <= 1))

    # Survival must be non-increasing
    assert np.all(np.diff(s) <= 1e-12)


def test_rsf_multiple_predictions():
    X, T, E, _ = load_waltons_dataset()

    model = RandomSurvivalForest(
        n_estimators=3,
        max_depth=3,
        min_samples_split=5,
        max_features=1,
    )
    model.fit(X, T, E)

    samples = np.array([[1.0, 0.0], [0.0, 1.0]])
    results = [model.predict_survival_function(x) for x in samples]

    # Ensure each prediction returns a valid curve
    for t, s in results:
        assert len(t) == len(s)
        assert np.all((s >= 0) & (s <= 1))
