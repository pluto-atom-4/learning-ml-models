import numpy as np
import pandas as pd
from machine_learning.model_comparison import (
    load_dataset,
    train_and_evaluate_models,
)

def test_load_dataset(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x,y\n1,2\n3,4\n")

    df = load_dataset(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y"]


def test_train_and_evaluate_models(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text(
        "x,y\n1,2\n2,4\n3,6\n4,8\n5,10\n6,12\n7,14\n8,16\n"
    )

    errors = train_and_evaluate_models(str(csv_file), test_size=0.25)

    assert len(errors) == 5
    assert all(isinstance(e, float) for e in errors)
