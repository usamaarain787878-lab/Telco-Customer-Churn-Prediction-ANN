# batch_prediction.py

import pandas as pd


def check_required_columns(df, required_columns):
    """
    Check whether required columns exist in uploaded CSV.
    """

    missing_columns = []

    for column in required_columns:

        if column not in df.columns:
            missing_columns.append(column)

    return missing_columns


def prepare_batch_data(df):
    """
    Basic preparation for uploaded customer data.
    """

    data = df.copy()

    # Remove spaces from column names
    data.columns = data.columns.str.strip()

    # Convert numeric columns if available
    numeric_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


def add_risk_level(probability):
    """
    Convert churn probability into risk level.
    """

    if probability < 0.30:

        return "Low"

    elif probability < 0.60:

        return "Medium"

    elif probability < 0.80:

        return "High"

    else:

        return "Critical"


def create_prediction_result(
    df,
    probabilities
):
    """
    Create final batch prediction results.
    """

    result = df.copy()

    result["Churn_Probability"] = (
        probabilities * 100
    ).round(2)

    result["Risk_Level"] = result[
        "Churn_Probability"
    ].apply(
        lambda x: add_risk_level(
            x / 100
        )
    )

    result["Prediction"] = result[
        "Churn_Probability"
    ].apply(
        lambda x:
        "Likely to Churn"
        if x >= 50
        else "Likely to Stay"
    )

    return result


# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":

    print("================================")
    print("BATCH CUSTOMER CHURN PREDICTION")
    print("================================")

    # Example customer data
    data = {
        "customerID": [
            "CUST001",
            "CUST002",
            "CUST003"
        ],

        "tenure": [
            5,
            48,
            12
        ],

        "MonthlyCharges": [
            85.50,
            55.20,
            92.10
        ]
    }

    df = pd.DataFrame(data)

    # Example probabilities
    # Later these will come from the ANN model
    probabilities = pd.Series([
        0.82,
        0.18,
        0.67
    ])

    result = create_prediction_result(
        df,
        probabilities
    )

    print("\nPrediction Results:")
    print(result)

    # Save result
    result.to_csv(
        "batch_prediction_results.csv",
        index=False
    )

    print(
        "\nResults saved as:"
    )

    print(
        "batch_prediction_results.csv"
    )
