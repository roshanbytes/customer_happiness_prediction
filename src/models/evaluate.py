import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.config import TARGET_ACCURACY


def evaluate_on_test_set(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def print_report(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    print(f"{model_name}")
    print(classification_report(y_test, y_pred, target_names=["unhappy", "happy"]))
    beat_target = accuracy_score(y_test, y_pred) >= TARGET_ACCURACY
    print(f"beat {TARGET_ACCURACY:.0%} target: {beat_target}")


def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["unhappy", "happy"], cmap="Blues"
    )
    plt.title(f"{model_name}'s confusion matrix")
    plt.show()


def plot_feature_importance(model, feature_names, model_name="Model"):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances.sort_values().plot(kind="barh")
    plt.title(f"{model_name}'s feature importance")
    plt.show()
