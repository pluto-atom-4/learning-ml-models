"""
Example: Binary Classification Model Selection using ROC and AUC

This script demonstrates how to use the model_selection_roc_auc module
to compare Logistic Regression and Random Forest models using ROC curves
and AUC scores.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from machine_learning.model_selection_roc_auc import (
    compute_roc_curve,
    compute_auc,
    train_logistic_regression,
    train_random_forest,
    evaluate_model_with_roc_auc,
    compare_models,
    print_model_comparison,
)


def main():
    """Main example function."""
    print("\n" + "=" * 70)
    print("Binary Classification Model Selection: ROC/AUC Analysis")
    print("=" * 70)

    # Step 1: Generate synthetic dataset
    print("\n[Step 1] Generating synthetic binary classification dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    print(f"  Dataset shape: {X.shape}")
    print(f"  Class distribution: {np.unique(y, return_counts=True)}")

    # Step 2: Split data into train/test
    print("\n[Step 2] Splitting data into train/test sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    print(f"  Training set: {x_train.shape}")
    print(f"  Test set: {x_test.shape}")

    # Step 3: Train and evaluate models
    print("\n[Step 3] Training models...")
    results = compare_models(x_train, y_train, x_test, y_test)

    # Step 4: Print comparison
    print_model_comparison(results)

    # Step 5: Detailed analysis
    print("\n[Step 5] Detailed ROC/AUC Analysis:")
    print("-" * 70)

    for model_name, data in results.items():
        metrics = data['metrics']
        fpr_test = metrics['fpr_test']
        tpr_test = metrics['tpr_test']
        auc_test = metrics['auc_test']

        print(f"\n{model_name}:")
        print(f"  Test AUC: {auc_test:.4f}")
        print(f"  FPR range: [{fpr_test[0]:.4f}, {fpr_test[-1]:.4f}]")
        print(f"  TPR range: [{tpr_test[0]:.4f}, {tpr_test[-1]:.4f}]")
        print(f"  ROC curve points: {len(fpr_test)}")

    # Step 6: Determine best model
    print("\n[Step 6] Model Selection:")
    print("-" * 70)
    best_model_name = max(
        results.items(),
        key=lambda x: x[1]['metrics']['auc_test']
    )[0]
    best_auc = results[best_model_name]['metrics']['auc_test']
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ Test AUC: {best_auc:.4f}")

    return results


if __name__ == '__main__':
    results = main()
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70 + "\n")

