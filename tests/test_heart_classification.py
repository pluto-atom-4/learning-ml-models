import numpy as np
import pandas as pd
from src.machine_learning.heart_classification import (
    load_heart_dataset,
    split_data,
    fit_knn,
    fit_logistic,
    evaluate_model,
)


def test_load_heart_dataset(tmp_path):
    csv_file = tmp_path / "heart.csv"
    csv_file.write_text(
        '"","Age","AHD"\n'
        '"1",63,"Yes"\n'
        '"2",50,"No"\n'
    )

    df = load_heart_dataset(str(csv_file))
    assert list(df.columns) == ["", "Age", "AHD"]
    assert df["AHD"].tolist() == [1, 0]


def test_split_data():
    df = pd.DataFrame({"Age": [30, 40, 50, 60], "AHD": [0, 1, 0, 1]})
    X_train, X_val, y_train, y_val = split_data(df, test_size=0.5, random_state=0)

    assert len(X_train) == 2
    assert len(X_val) == 2


def test_knn_and_logistic():
    df = pd.DataFrame({"Age": [30, 40, 50, 60], "AHD": [0, 1, 0, 1]})
    X_train, X_val, y_train, y_val = split_data(df, test_size=0.5, random_state=0)

    knn = fit_knn(X_train, y_train, k=1)
    log = fit_logistic(X_train, y_train)

    train_acc_knn, val_acc_knn = evaluate_model(knn, X_train, y_train, X_val, y_val)
    train_acc_log, val_acc_log = evaluate_model(log, X_train, y_train, X_val, y_val)

    assert 0 <= train_acc_knn <= 1
    assert 0 <= val_acc_knn <= 1
    assert 0 <= train_acc_log <= 1
    assert 0 <= val_acc_log <= 1
