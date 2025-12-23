"""
Tests for ROC/AUC model selection module.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from machine_learning.model_selection_roc_auc import (
    compute_roc_curve,
    compute_auc,
    compute_auc_score,
    train_logistic_regression,
    train_random_forest,
    evaluate_model_with_roc_auc,
    compare_models,
)


def test_compute_roc_curve():
    """Test ROC curve computation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8])

    fpr, tpr, thresholds = compute_roc_curve(y_true, y_pred_proba)

    # Check shape consistency
    assert len(fpr) == len(tpr), "FPR and TPR should have same length"
    assert fpr[0] == 0.0, "FPR should start at 0"
    assert tpr[0] == 0.0, "TPR should start at 0"
    assert fpr[-1] <= 1.0, "FPR should not exceed 1"
    assert tpr[-1] <= 1.0, "TPR should not exceed 1"

    print("✓ test_compute_roc_curve passed")


def test_compute_auc():
    """Test AUC computation."""
    # Perfect classifier
    fpr_perfect = np.array([0.0, 0.0, 1.0])
    tpr_perfect = np.array([0.0, 1.0, 1.0])
    auc_perfect = compute_auc(fpr_perfect, tpr_perfect)
    assert auc_perfect == 1.0, "Perfect classifier should have AUC=1.0"

    # Random classifier
    fpr_random = np.array([0.0, 0.5, 1.0])
    tpr_random = np.array([0.0, 0.5, 1.0])
    auc_random = compute_auc(fpr_random, tpr_random)
    assert 0.45 < auc_random < 0.55, "Random classifier should have AUC≈0.5"

    print("✓ test_compute_auc passed")


def test_compute_auc_score():
    """Test direct AUC score computation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8])

    auc = compute_auc_score(y_true, y_pred_proba)
    assert 0.0 <= auc <= 1.0, "AUC should be between 0 and 1"

    print("✓ test_compute_auc_score passed")


def test_train_logistic_regression():
    """Test Logistic Regression training."""
    # Simple binary classification dataset
    x_train, x_test, y_train, y_test = train_test_split(
        *make_classification(n_samples=100, n_features=5, n_informative=3,
                             n_redundant=0, n_classes=2, random_state=42),
        test_size=0.3,
        random_state=42
    )

    model = train_logistic_regression(x_train, y_train)

    # Check model predictions
    y_pred = model.predict(x_test)
    assert len(y_pred) == len(y_test), "Predictions should match test set size"
    assert set(y_pred).issubset({0, 1}), "Predictions should be binary"

    # Check probability predictions
    y_proba = model.predict_proba(x_test)
    assert y_proba.shape == (len(y_test), 2), "Probabilities should be (n_samples, 2)"
    assert np.allclose(y_proba.sum(axis=1), 1.0), "Probabilities should sum to 1"

    print("✓ test_train_logistic_regression passed")


def test_train_random_forest():
    """Test Random Forest training."""
    x_train, x_test, y_train, y_test = train_test_split(
        *make_classification(n_samples=100, n_features=5, n_informative=3,
                             n_redundant=0, n_classes=2, random_state=42),
        test_size=0.3,
        random_state=42
    )

    model = train_random_forest(x_train, y_train)

    # Check model predictions
    y_pred = model.predict(x_test)
    assert len(y_pred) == len(y_test), "Predictions should match test set size"
    assert set(y_pred).issubset({0, 1}), "Predictions should be binary"

    # Check probability predictions
    y_proba = model.predict_proba(x_test)
    assert y_proba.shape == (len(y_test), 2), "Probabilities should be (n_samples, 2)"
    assert np.allclose(y_proba.sum(axis=1), 1.0), "Probabilities should sum to 1"

    print("✓ test_train_random_forest passed")


def test_evaluate_model_with_roc_auc():
    """Test model evaluation with ROC/AUC."""
    x_train, x_test, y_train, y_test = train_test_split(
        *make_classification(n_samples=100, n_features=5, n_informative=3,
                             n_redundant=0, n_classes=2, random_state=42),
        test_size=0.3,
        random_state=42
    )

    model = train_logistic_regression(x_train, y_train)
    results = evaluate_model_with_roc_auc(model, x_train, y_train, x_test, y_test)

    # Check results structure
    assert 'auc_train' in results
    assert 'auc_test' in results
    assert 'fpr_train' in results
    assert 'tpr_train' in results
    assert 'fpr_test' in results
    assert 'tpr_test' in results

    # Check AUC values are valid
    assert 0.0 <= results['auc_train'] <= 1.0
    assert 0.0 <= results['auc_test'] <= 1.0

    print("✓ test_evaluate_model_with_roc_auc passed")


def test_compare_models():
    """Test model comparison."""
    x_train, x_test, y_train, y_test = train_test_split(
        *make_classification(n_samples=100, n_features=5, n_informative=3,
                             n_redundant=0, n_classes=2, random_state=42),
        test_size=0.3,
        random_state=42
    )

    results = compare_models(x_train, y_train, x_test, y_test)

    # Check both models are included
    assert 'Logistic Regression' in results
    assert 'Random Forest' in results

    # Check results structure
    for model_name, data in results.items():
        assert 'model' in data
        assert 'metrics' in data
        assert 'auc_test' in data['metrics']
        assert 0.0 <= data['metrics']['auc_test'] <= 1.0

    print("✓ test_compare_models passed")


if __name__ == '__main__':
    print("\nRunning ROC/AUC model selection tests...\n")

    test_compute_roc_curve()
    test_compute_auc()
    test_compute_auc_score()
    test_train_logistic_regression()
    test_train_random_forest()
    test_evaluate_model_with_roc_auc()
    test_compare_models()

    print("\n✅ All tests passed!\n")

