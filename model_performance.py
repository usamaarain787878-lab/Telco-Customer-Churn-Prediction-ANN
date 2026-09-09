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


def calculate_metrics(y_true, y_pred, y_probability=None):
    """
    Calculate machine learning model performance metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # ROC-AUC
    if y_probability is not None:

        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )

    else:

        roc_auc = None

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


def create_confusion_matrix(y_true, y_pred):
    """
    Create confusion matrix chart.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(
            x="Predicted",
            y="Actual",
            color="Customers"
        ),
        x=["No Churn", "Churn"],
        y=["No Churn", "Churn"],
        title="Confusion Matrix"
    )

    return fig


def create_metrics_dataframe(metrics):
    """
    Convert metrics dictionary into DataFrame.
    """

    data = {
        "Metric": [],
        "Score": []
    }

    for name, value in metrics.items():

        data["Metric"].append(name)

        if value is not None:

            data["Score"].append(
                round(value * 100, 2)
            )

        else:

            data["Score"].append(None)

    return pd.DataFrame(data)


# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":

    # Example actual values
    y_true = [
        0, 1, 0, 1, 1,
        0, 0, 1, 0, 1
    ]

    # Example model predictions
    y_pred = [
        0, 1, 0, 1, 0,
        0, 1, 1, 0, 1
    ]

    # Example probabilities
    y_probability = [
        0.10,
        0.90,
        0.20,
        0.85,
        0.40,
        0.15,
        0.70,
        0.80,
        0.25,
        0.95
    ]

    # Calculate metrics
    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probability
    )

    print("================================")
    print("ANN MODEL PERFORMANCE")
    print("================================")

    for name, value in metrics.items():

        if value is not None:

            print(
                f"{name}: {value * 100:.2f}%"
            )

        else:

            print(
                f"{name}: Not Available"
            )

    # Create DataFrame
    metrics_df = create_metrics_dataframe(
        metrics
    )

    print("\nMetrics Summary:")
    print(metrics_df)

    # Confusion Matrix
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )
