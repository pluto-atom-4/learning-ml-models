import numpy as np
import pandas as pd
from src.machine_learning.bias_variance_polynomial import (
    load_noisy_population,
    sample_dataset,
    compute_polynomial_predictions,
)


def test_load_noisy_population(tmp_path):
    csv_file = tmp_path / "noisy.csv"
    csv_file.write_text("f,x,y\n0.1,0.0,1.0\n0.2,0.1,2.0\n")

    f, x, y = load_noisy_population(str(csv_file))

    assert np.allclose(f, [0.1, 0.2])
    assert np.allclose(x, [0.0, 0.1])
    assert np.allclose(y, [1.0, 2.0])


def test_sample_dataset():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 20, 30, 40, 50])

    xs, ys = sample_dataset(x, y, sample_size=3, random_state=0)

    assert len(xs) == 3
    assert len(ys) == 3


def test_compute_polynomial_predictions():
    x = np.linspace(0, 1, 10)
    y = x**2

    degrees = [1, 2, 3]
    predictions, x_pred = compute_polynomial_predictions(x, y, degrees)

    assert set(predictions.keys()) == {1, 2, 3}
    assert len(x_pred) == 100
    for deg in degrees:
        assert len(predictions[deg]) == 100
