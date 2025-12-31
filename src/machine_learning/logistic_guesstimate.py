from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


def load_insurance_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    return df["age"].to_numpy(), df["insuranceclaim"].to_numpy()


def logistic(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


@dataclass
class ResultNode:
    beta0: float
    beta1: float
    accuracy: float
    next: ResultNode | None = None


def insert_sorted(head: ResultNode | None, node: ResultNode) -> ResultNode:
    """Insert into linked list sorted by accuracy (descending)."""
    if head is None or node.accuracy > head.accuracy:
        node.next = head
        return node

    current = head
    while current.next and current.next.accuracy >= node.accuracy:
        current = current.next

    node.next = current.next
    current.next = node
    return head


def guesstimate_logistic_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int = 8,
    threshold: float = 0.5,
) -> ResultNode | None:

    head: ResultNode | None = None
    rng = np.random.default_rng()

    for _ in range(iterations):
        idx = rng.choice(len(x), size=2, replace=False)
        x1, x2 = x[idx]
        y1, y2 = y[idx]

        if x1 == x2:
            continue

        beta1 = (y2 - y1) / (x2 - x1)
        beta0 = y1 - beta1 * x1

        probs = logistic(beta0 + beta1 * x)
        y_pred = (probs >= threshold).astype(int)
        accuracy = float((y_pred == y).mean())

        node = ResultNode(beta0=beta0, beta1=beta1, accuracy=accuracy)
        head = insert_sorted(head, node)

    return head
