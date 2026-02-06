from sklearn.neighbors import KNeighborsClassifier

#from model_evaluation import (    load_and_prepare_datasets, report_results )
#from model_evaluation_viz import show_confusion_matrix_plot

try:
    # When executed as part of the package
    from .model_evaluation import load_and_prepare_datasets, report_results
    from .model_evaluation_viz import show_confusion_matrix_plot
except ImportError:
    # When executed directly: python model_evaluation_runner.py
    from machine_learning.exploratory_data_analysis.model_evaluation import load_and_prepare_datasets, report_results
    from machine_learning.exploratory_data_analysis.model_evaluation_viz import show_confusion_matrix_plot


def report_and_visualize_results(y_test, y_pred, show_plot=True):
    """
    Output classification report, confusion matrix, and optionally display confusion matrix plot.

    Args:
        y_test: Actual target values
        y_pred: Predicted target values
        show_plot (bool): Whether to display the confusion matrix plot (default: True)
    """
    # Output text-based reports
    report_results(y_test, y_pred)

    # Visualize the Confusion Matrix if requested
    if show_plot:
        show_confusion_matrix_plot(y_test, y_pred)

def run_prediction_pipeline():
    """
    Executes the full prediction pipeline: data loading, model training, prediction, and result reporting/visualization.
    """

    X_train, y_train, X_test, y_test = load_and_prepare_datasets()
    if X_train is None:
        return

    # Initialize and Fit kNN (using k=5 as a baseline)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Report and visualize results with plot display
    report_and_visualize_results(y_test, y_pred, show_plot=True)

if __name__ == "__main__":
    run_prediction_pipeline()
