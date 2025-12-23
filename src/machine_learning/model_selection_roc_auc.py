"""
Model Selection using ROC and AUC for Binary Classification.

This module compares Logistic Regression and Random Forest models
using ROC (Receiver Operating Characteristic) curves and AUC
(Area Under the Curve) scores.

Pure functional implementations demonstrate the concepts while
scikit-learn methods provide production-ready alternatives.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ============================================================================
# CUSTOM IMPLEMENTATIONS: ROC/AUC from scratch
# ============================================================================

def compute_roc_curve(y_true, y_pred_proba):
    """
    Compute ROC curve (True Positive Rate vs False Positive Rate).

    Parameters:
        y_true: array-like, true binary labels (0, 1)
        y_pred_proba: array-like, predicted probabilities for class 1

    Returns:
        fpr: array, false positive rates
        tpr: array, true positive rates
        thresholds: array, classification thresholds
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Sort by probability (descending)
    sorted_indices = np.argsort(-y_pred_proba)
    y_true_sorted = y_true[sorted_indices]

    # Get unique thresholds
    unique_thresholds = np.unique(y_pred_proba)
    thresholds = np.sort(unique_thresholds)[::-1]

    # Total positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    fpr = []
    tpr = []

    # For each threshold, compute TPR and FPR
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        # Handle division by zero
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr.append(tpr_val)
        fpr.append(fpr_val)

    # Add point (0, 0) at the beginning
    fpr = [0] + fpr
    tpr = [0] + tpr

    return np.array(fpr), np.array(tpr), thresholds


def compute_auc(fpr, tpr):
    """
    Compute Area Under the ROC Curve using trapezoid rule.

    Parameters:
        fpr: array, false positive rates
        tpr: array, true positive rates

    Returns:
        auc: float, area under the curve (0 to 1)
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    # Trapezoid rule: sum of rectangles
    auc = 0.0
    for i in range(len(fpr) - 1):
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i] + tpr[i + 1]) / 2.0

    return auc


def compute_auc_score(y_true, y_pred_proba):
    """
    Compute AUC score directly from true labels and predicted probabilities.

    Parameters:
        y_true: array-like, true binary labels (0, 1)
        y_pred_proba: array-like, predicted probabilities for class 1

    Returns:
        auc: float, AUC score (0 to 1)
    """
    fpr, tpr, _ = compute_roc_curve(y_true, y_pred_proba)
    return compute_auc(fpr, tpr)


# ============================================================================
# MODEL SELECTION WITH LOGISTIC REGRESSION & RANDOM FOREST
# ============================================================================

def train_logistic_regression(x_train, y_train, random_state=42):
    """
    Train a Logistic Regression model for binary classification.

    Parameters:
        x_train: array-like, training features
        y_train: array-like, training labels (0, 1)
        random_state: int, random seed for reproducibility

    Returns:
        model: trained LogisticRegression object
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs'
    )
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest model for binary classification.

    Parameters:
        x_train: array-like, training features
        y_train: array-like, training labels (0, 1)
        n_estimators: int, number of trees in the forest
        random_state: int, random seed for reproducibility

    Returns:
        model: trained RandomForestClassifier object
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=10
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model_with_roc_auc(model, x_train, y_train, x_test, y_test):
    """
    Evaluate a binary classification model using ROC and AUC.

    Parameters:
        model: trained classifier with predict_proba method
        x_train: array-like, training features
        y_train: array-like, training labels
        x_test: array-like, test features
        y_test: array-like, test labels

    Returns:
        results: dict with train/test AUC scores and ROC curve data
    """
    # Get predicted probabilities
    y_train_proba = model.predict_proba(x_train)[:, 1]
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # Compute ROC curves and AUC scores
    fpr_train, tpr_train, _ = compute_roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = compute_roc_curve(y_test, y_test_proba)

    auc_train = compute_auc(fpr_train, tpr_train)
    auc_test = compute_auc(fpr_test, tpr_test)

    return {
        'auc_train': auc_train,
        'auc_test': auc_test,
        'fpr_train': fpr_train,
        'tpr_train': tpr_train,
        'fpr_test': fpr_test,
        'tpr_test': tpr_test,
    }


def compare_models(x_train, y_train, x_test, y_test, random_state=42):
    """
    Train and compare Logistic Regression and Random Forest models.

    Parameters:
        x_train: array-like, training features
        y_train: array-like, training labels (0, 1)
        x_test: array-like, test features
        y_test: array-like, test labels (0, 1)
        random_state: int, random seed for reproducibility

    Returns:
        results: dict with model names and their evaluation metrics
    """
    results = {}

    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(x_train, y_train, random_state)
    lr_results = evaluate_model_with_roc_auc(lr_model, x_train, y_train, x_test, y_test)
    results['Logistic Regression'] = {
        'model': lr_model,
        'metrics': lr_results
    }

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(x_train, y_train, random_state=random_state)
    rf_results = evaluate_model_with_roc_auc(rf_model, x_train, y_train, x_test, y_test)
    results['Random Forest'] = {
        'model': rf_model,
        'metrics': rf_results
    }

    return results


def print_model_comparison(results):
    """
    Print comparison summary of models.

    Parameters:
        results: dict with model evaluation results
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: ROC/AUC Scores")
    print("=" * 70)

    for model_name, data in results.items():
        metrics = data['metrics']
        print(f"\n{model_name}:")
        print(f"  Training AUC:   {metrics['auc_train']:.4f}")
        print(f"  Test AUC:       {metrics['auc_test']:.4f}")
        print(f"  Generalization: {metrics['auc_train'] - metrics['auc_test']:.4f}")

    # Determine best model
    best_model = max(
        results.items(),
        key=lambda x: x[1]['metrics']['auc_test']
    )
    print(f"\n{'Best Model (by Test AUC):':<20} {best_model[0]}")
    print(f"{'Test AUC:':<20} {best_model[1]['metrics']['auc_test']:.4f}")
    print("=" * 70)


# ============================================================================
# SKLEARN REFERENCE IMPLEMENTATIONS
# ============================================================================

def compute_auc_sklearn(y_true, y_pred_proba):
    """
    Compute AUC score using scikit-learn (reference implementation).

    Parameters:
        y_true: array-like, true binary labels (0, 1)
        y_pred_proba: array-like, predicted probabilities for class 1

    Returns:
        auc: float, AUC score
    """
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred_proba)


def compute_roc_curve_sklearn(y_true, y_pred_proba):
    """
    Compute ROC curve using scikit-learn (reference implementation).

    Parameters:
        y_true: array-like, true binary labels (0, 1)
        y_pred_proba: array-like, predicted probabilities for class 1

    Returns:
        fpr: array, false positive rates
        tpr: array, true positive rates
        thresholds: array, classification thresholds
    """
    from sklearn.metrics import roc_curve
    return roc_curve(y_true, y_pred_proba)

