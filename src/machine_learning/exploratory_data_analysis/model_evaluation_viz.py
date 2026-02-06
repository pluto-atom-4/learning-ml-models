import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix: COVID-19 Hospitalization Urgency",
                         display_labels=None, figsize=(8, 6), cmap='Blues'):
    """
    Create and return a confusion matrix visualization without displaying.

    Args:
        y_test: Actual target values
        y_pred: Predicted target values
        title (str): Title for the plot
        display_labels (list): Labels for the confusion matrix axes (default: ['No Urgency', 'High Urgency'])
        figsize (tuple): Figure size in inches (default: (8, 6))
        cmap (str): Colormap for the heatmap (default: 'Blues')

    Returns:
        tuple: (fig, ax, cm) – Figure object, axes object, and confusion matrix
    """
    if display_labels is None:
        display_labels = ['No Urgency', 'High Urgency']

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=cmap, ax=ax)
    plt.title(title)

    return fig, ax, cm


def show_confusion_matrix_plot(y_test, y_pred, title="Confusion Matrix: COVID-19 Hospitalization Urgency",
                               display_labels=None, figsize=(8, 6), cmap='Blues'):
    """
    Create, display, and show a confusion matrix visualization.

    Args:
        y_test: Actual target values
        y_pred: Predicted target values
        title (str): Title for the plot
        display_labels (list): Labels for the confusion matrix axes (default: ['No Urgency', 'High Urgency'])
        figsize (tuple): Figure size in inches (default: (8, 6))
        cmap (str): Colormap for the heatmap (default: 'Blues')

    Returns:
        tuple: (fig, ax, cm) – Figure object, axes object, and confusion matrix
    """
    fig, ax, cm = plot_confusion_matrix(y_test, y_pred, title=title,
                                       display_labels=display_labels,
                                       figsize=figsize, cmap=cmap)
    plt.show()
    return fig, ax, cm


def save_confusion_matrix_plot(y_test, y_pred, filepath, title="Confusion Matrix: COVID-19 Hospitalization Urgency",
                               display_labels=None, figsize=(8, 6), cmap='Blues', dpi=300):
    """
    Create and save a confusion matrix visualization to file.

    Args:
        y_test: Actual target values
        y_pred: Predicted target values
        filepath (str): Path where the plot should be saved
        title (str): Title for the plot
        display_labels (list): Labels for the confusion matrix axes (default: ['No Urgency', 'High Urgency'])
        figsize (tuple): Figure size in inches (default: (8, 6))
        cmap (str): Colormap for the heatmap (default: 'Blues')
        dpi (int): DPI for saved image (default: 300)

    Returns:
        np.ndarray: The confusion matrix array
    """
    fig, ax, cm = plot_confusion_matrix(y_test, y_pred, title=title,
                                       display_labels=display_labels,
                                       figsize=figsize, cmap=cmap)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {filepath}")
    return cm
