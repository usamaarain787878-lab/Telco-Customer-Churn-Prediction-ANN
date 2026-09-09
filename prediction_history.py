import pandas as pd
import os
from datetime import datetime


HISTORY_FILE = "prediction_history.csv"


def save_prediction(customer_id, churn_probability, risk_level, prediction):
    """
    Save one customer prediction into prediction history.
    """

    new_record = {
        "Customer_ID": customer_id,
        "Prediction_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Churn_Probability": round(churn_probability, 2),
        "Risk_Level": risk_level,
        "Prediction": prediction
    }

    new_data = pd.DataFrame([new_record])

    # If history file already exists, add new record
    if os.path.exists(HISTORY_FILE):
        old_data = pd.read_csv(HISTORY_FILE)
        final_data = pd.concat([old_data, new_data], ignore_index=True)
    else:
        final_data = new_data

    final_data.to_csv(HISTORY_FILE, index=False)

    return "Prediction saved successfully."


def load_prediction_history():
    """
    Load all previous prediction records.
    """

    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame(
        columns=[
            "Customer_ID",
            "Prediction_Date",
            "Churn_Probability",
            "Risk_Level",
            "Prediction"
        ]
    )


# --------------------------------------------------
# Example
# --------------------------------------------------

if __name__ == "__main__":

    message = save_prediction(
        customer_id="CUST001",
        churn_probability=82.5,
        risk_level="Critical",
        prediction="Likely to Churn"
    )

    print(message)

    history = load_prediction_history()

    print("\nPrediction History:")
    print(history)
