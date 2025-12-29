import pandas as pd
import numpy as np


def compute_rfm(df, customer_col, date_col, amount_col, reference_date=None):
    """
    Compute Recency, Frequency, Monetary (RFM) metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Transaction data.
    customer_col : str
        Column name for customer ID.
    date_col : str
        Column name for transaction date.
    amount_col : str
        Column name for transaction amount.
    reference_date : datetime or None
        If None, uses max(date_col) in df.

    Returns
    -------
    rfm_df : pandas.DataFrame
        DataFrame with columns: Recency, Frequency, Monetary.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if reference_date is None:
        reference_date = df[date_col].max()

    # Group by customer
    grouped = df.groupby(customer_col)

    recency = (reference_date - grouped[date_col].max()).dt.days
    frequency = grouped[date_col].count()
    monetary = grouped[amount_col].sum()

    rfm_df = pd.DataFrame({
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    })

    return rfm_df


def score_rfm(rfm_df):
    """
    Score RFM metrics using quartiles (1–4 scale).
    Higher score = better customer.

    Returns
    -------
    scored_df : pandas.DataFrame
        RFM with additional columns: R_score, F_score, M_score, RFM_score.
    """
    scored = rfm_df.copy()

    # Recency: lower is better → reverse scoring
    scored["R_score"] = pd.qcut(scored["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)

    # Frequency: higher is better
    scored["F_score"] = pd.qcut(scored["Frequency"].rank(method="first"), 4,
                                labels=[1, 2, 3, 4]).astype(int)

    # Monetary: higher is better
    scored["M_score"] = pd.qcut(scored["Monetary"], 4,
                                labels=[1, 2, 3, 4]).astype(int)

    scored["RFM_score"] = scored["R_score"] + scored["F_score"] + scored["M_score"]

    return scored
