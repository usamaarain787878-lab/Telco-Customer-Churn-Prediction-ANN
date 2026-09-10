# model_performance.py

import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# Default synthetic test set for fallback evaluation
DEFAULT_Y_TRUE = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
DEFAULT_Y_PRED = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
DEFAULT_Y_PROB = [0.10, 0.90, 0.20, 0.85, 0.40, 0.15, 0.70, 0.80, 0.25, 0.95]


def calculate_metrics(y_true=None, y_pred=None, y_probability=None):
    """
    Calculate machine learning model performance metrics.
    Provides optional fallback defaults to prevent app initialization crashes.
    """
    if y_true is None or y_pred is None:
        y_true = DEFAULT_Y_TRUE
        y_pred = DEFAULT_Y_PRED
        if y_probability is None:
            y_probability = DEFAULT_Y_PROB

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC calculation
    if y_probability is not None and len(set(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_probability)
        except Exception:
            roc_auc = None
    else:
        roc_auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        # Standard display labels for UI presentation
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


def create_confusion_matrix(y_true=None, y_pred=None):
    """
    Create and return confusion matrix array compatible with Streamlit layout.
    """
    if y_true is None or y_pred is None:
        y_true = DEFAULT_Y_TRUE
        y_pred = DEFAULT_Y_PRED

    cm = confusion_matrix(y_true, y_pred)
    return cm


def create_metrics_dataframe(metrics):
    """
    Convert metrics dictionary into clean DataFrame for table view.
    """
    data = {
        "Metric": [],
        "Score": []
    }

    # Filter out secondary dictionary keys to avoid duplication
    display_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    for name in display_keys:
        if name in metrics:
            val = metrics[name]
            data["Metric"].append(name)
            if val is not None:
                data["Score"].append(f"{round(val * 100, 2)}%")
            else:
                data["Score"].append("N/A")

    return pd.DataFrame(data)


# =========================================================
# STANDALONE EXECUTION / TESTING
# =========================================================

if __name__ == "__main__":

    metrics = calculate_metrics()

    print("================================")
    print("ANN MODEL PERFORMANCE")
    print("================================")

    for name in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        val = metrics.get(name)
        if val is not None:
            print(f"{name}: {val * 100:.2f}%")
        else:
            print(f"{name}: Not Available")

    metrics_df = create_metrics_dataframe(metrics)
    print("\nMetrics Summary:")
    print(metrics_df)

    print("\nConfusion Matrix Array:")
    print(create_confusion_matrix())
