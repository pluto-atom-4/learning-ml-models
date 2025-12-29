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
from visualizations.plot_heart_classification import plot_overlapped_classification_results, plot_smooth_decision_boundary, plot_overlapped_smooth_boundaries


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
    print("Training k-NN classifier (k=20)...")
    knn = fit_knn(X_train, y_train, k=20)

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

    # Generate predictions and probabilities
    print("Generating predictions and probabilities...")
    knn_pred = knn.predict(X_val)
    log_pred = log.predict(X_val)

    # Get probability estimates for each class
    knn_proba = knn.predict_proba(X_val)
    log_proba = log.predict_proba(X_val)

    print(f"kNN prediction samples (first 5): {knn_pred[:5]}")
    print(f"kNN probability samples (first 5):\n{knn_proba[:5]}")
    print(f"Logistic prediction samples (first 5): {log_pred[:5]}")
    print(f"Logistic probability samples (first 5):\n{log_proba[:5]}\n")

    # Visualize all results in one overlapped canvas
    print("Generating overlapped multi-layer visualization with all plots...")
    plot_overlapped_classification_results(X_train, y_train, X_val, y_val, knn_pred, log_pred, knn_proba, log_proba)
    print("Overlapped visualization complete!")

    # Visualize smooth decision boundaries and probability curves
    print("\nGenerating smooth decision boundary visualization with linspace...")
    plot_smooth_decision_boundary(X_train, y_train, X_val, y_val, knn, log)
    print("Smooth decision boundary visualization complete!")

    # Visualize overlapped smooth boundaries with linspace dummy data
    print("\nGenerating overlapped smooth boundaries visualization with linspace...")
    plot_overlapped_smooth_boundaries(X_train, y_train, X_val, y_val, knn, log)
    print("Overlapped smooth boundaries visualization complete!")


if __name__ == "__main__":
    main()
