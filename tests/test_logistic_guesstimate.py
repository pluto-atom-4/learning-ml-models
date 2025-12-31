from pathlib import Path
import pandas as pd
import numpy as np
from src.machine_learning.logistic_guesstimate import (
    load_insurance_data,
    guesstimate_logistic_coefficients,
    ResultNode,
)


def test_load_insurance_data(tmp_path: Path):
    csv_file = tmp_path / "insurance_claim.csv"
    csv_file.write_text("age,insuranceclaim\n20,0\n30,1\n")

    x, y = load_insurance_data(csv_file)

    assert np.allclose(x, [20, 30])
    assert np.allclose(y, [0, 1])


def test_guesstimate_logistic_coefficients(tmp_path: Path):
    csv_file = tmp_path / "insurance_claim.csv"
    csv_file.write_text(
        "age,insuranceclaim\n"
        "20,0\n25,0\n30,1\n35,1\n"
    )

    x, y = load_insurance_data(csv_file)
    head = guesstimate_logistic_coefficients(x, y, iterations=8)

    assert isinstance(head, ResultNode)

    accuracies = []
    current = head
    while current:
        accuracies.append(current.accuracy)
        current = current.next

    assert len(accuracies) >= 1
    assert accuracies == sorted(accuracies, reverse=True)
