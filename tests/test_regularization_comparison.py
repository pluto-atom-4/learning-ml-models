import numpy as np
import pandas as pd
from src.machine_learning.regularization_comparison import (
    load_dataset,
    compare_regularization_models,
)


def test_load_dataset(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x1,x2,y\n1,2,3\n4,5,6\n")

    df = load_dataset(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x1", "x2", "y"]


def test_compare_regularization_models(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "x1,x2,y\n1,2,3\n2,3,5\n3,4,7\n4,5,9\n5,6,11\n"
    )

    coefficients, errors, feature_names = compare_regularization_models(
        str(csv_file), alpha=0.1
    )

    assert set(coefficients.keys()) == {"linear", "lasso", "ridge"}
    assert set(errors.keys()) == {"linear", "lasso", "ridge"}
    assert feature_names == ["x1", "x2"]

    for model in coefficients:
        assert len(coefficients[model]) == 2
        assert isinstance(errors[model], float)
