import pandas as pd
from src.machine_learning.rfm import compute_rfm, score_rfm


def test_compute_rfm_basic():
    df = pd.DataFrame({
        "CustomerID": [1, 1, 2],
        "Date": ["2024-01-01", "2024-01-10", "2024-01-05"],
        "Amount": [100, 200, 50]
    })

    rfm = compute_rfm(df, "CustomerID", "Date", "Amount",
                      reference_date=pd.Timestamp("2024-01-15"))

    assert rfm.loc[1, "Recency"] == 5      # last purchase Jan 10
    assert rfm.loc[1, "Frequency"] == 2
    assert rfm.loc[1, "Monetary"] == 300

    assert rfm.loc[2, "Recency"] == 10     # last purchase Jan 5
    assert rfm.loc[2, "Frequency"] == 1
    assert rfm.loc[2, "Monetary"] == 50


def test_score_rfm():
    rfm = pd.DataFrame({
        "Recency": [5, 20, 40, 60],
        "Frequency": [10, 5, 3, 1],
        "Monetary": [500, 200, 100, 50]
    })

    scored = score_rfm(rfm)

    assert "R_score" in scored.columns
    assert "F_score" in scored.columns
    assert "M_score" in scored.columns
    assert "RFM_score" in scored.columns

    assert scored["RFM_score"].min() >= 3
    assert scored["RFM_score"].max() <= 12
