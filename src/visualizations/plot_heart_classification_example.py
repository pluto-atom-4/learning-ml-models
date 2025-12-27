"""
Heart Classification Example Script

This module demonstrates k-NN and Logistic Regression classification
on the Heart dataset, comparing their performance and visualizing results.
"""

from machine_learning.heart_classification import (
    load_heart_dataset,
    split_data,
    fit_knn,
    fit_logistic,
    evaluate_model,
)
from visualizations.plot_heart_classification import plot_classification_results


def main():
    """
    Main function to run heart classification example.

    Loads the Heart dataset, trains k-NN and Logistic Regression models,
    evaluates both on training and validation sets, and visualizes results.
    """
    # Load and prepare data
    csv_path = "../../generated/data/raw/Heart.csv"
    print(f"Loading dataset from: {csv_path}")

    df = load_heart_dataset(csv_path)
    X_train, X_val, y_train, y_val = split_data(df)
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}\n")

    # Train models
    print("Training k-NN classifier (k=5)...")
    knn = fit_knn(X_train, y_train, k=5)

    print("Training Logistic Regression classifier...")
    log = fit_logistic(X_train, y_train)

    # Evaluate models
    print("\nEvaluating models...")
    train_knn, val_knn = evaluate_model(knn, X_train, y_train, X_val, y_val)
    train_log, val_log = evaluate_model(log, X_train, y_train, X_val, y_val)

    # Display results
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"k-NN Accuracies       - Train: {train_knn:.4f}, Val: {val_knn:.4f}")
    print(f"Logistic Accuracies   - Train: {train_log:.4f}, Val: {val_log:.4f}")
    print("=" * 50 + "\n")

    # Generate predictions and visualize results
    print("Generating predictions and visualization...")
    knn_pred = knn.predict(X_val)
    log_pred = log.predict(X_val)

    plot_classification_results(X_train, y_train, X_val, knn_pred, log_pred)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
