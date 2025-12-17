import numpy as np
from src.machine_learning.f1_score import precision, recall, f1_score

def test_precision_basic():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    # tp = 1, fp = 1 → precision = 1/2
    assert precision(y_true, y_pred) == 0.5


def test_recall_basic():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    # tp = 1, fn = 1 → recall = 1/2
    assert recall(y_true, y_pred) == 0.5


def test_f1_score_basic():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    # precision = 0.5, recall = 0.5 → F1 = 0.5
    assert f1_score(y_true, y_pred) == 0.5


def test_f1_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert f1_score(y_true, y_pred) == 1.0


def test_f1_zero_case():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])
    assert f1_score(y_true, y_pred) == 0.0
