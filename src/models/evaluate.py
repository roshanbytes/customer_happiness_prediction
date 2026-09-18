import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)

from src.utils.config import FEATURE_DESCRIPTIONS, TARGET_ACCURACY


def print_report(model, X_test, y_test, model_name="Model"):
    """
    Print a per-class classification report and the target accuracy check.

    Parameters
    ----------
    model : fitted estimator
        A trained classifier with a .predict() method.
    X_test : array-like
        Held-out feature matrix.
    y_test : array-like
        True labels matching X_test.
    model_name : str, optional
        Label used in the printed header. Defaults to "Model".

    Returns
    -------
    None
        Prints the report and target check.
    """
    y_pred = model.predict(X_test)
    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=["unhappy", "happy"]))
    beat_target = accuracy_score(y_test, y_pred) >= TARGET_ACCURACY
    print(f"beat {TARGET_ACCURACY:.0%} target: {beat_target}")


def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    """
    Plot a confusion matrix with unhappy/happy class labels.

    Parameters
    ----------
    model : fitted estimator
        A trained classifier with a .predict() method.
    X_test : array-like
        Held-out feature matrix.
    y_test : array-like
        True labels matching X_test.
    model_name : str, optional
        Used in the chart title. Defaults to "Model".

    Returns
    -------
    None
        Displays the plot.
    """
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["unhappy", "happy"], cmap="Blues"
    )
    plt.title(f"{model_name}'s confusion matrix")
    plt.show()


def plot_feature_importance(model, feature_names, model_name="Model"):
    """
    Plot feature importances as a horizontal bar chart.

    Parameters
    ----------
    model : fitted estimator
        A trained classifier exposing .feature_importances_.
    feature_names : list of str
        Column codes matching the model's training columns.
        Each is translated to its survey question text via
        FEATURE_DESCRIPTIONS before being plotted.
    model_name : str, optional
        Used in the chart title. Defaults to "Model".

    Returns
    -------
    None
        Displays the plot.
    """
    labels = [FEATURE_DESCRIPTIONS.get(name, name) for name in feature_names]
    importances = pd.Series(model.feature_importances_, index=labels)
    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh")
    plt.title(f"{model_name}'s feature importance")
    plt.tight_layout()
    plt.show()
