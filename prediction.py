# prediction.py

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# =========================================================
# MODEL PATHS
# =========================================================

MODEL_PATH = os.path.join(
    "models",
    "churn_model.keras"
)

SCALER_PATH = os.path.join(
    "models",
    "scaler.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_churn_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = load_model(MODEL_PATH)

    return model


# =========================================================
# LOAD SCALER
# =========================================================

def load_scaler():

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return scaler


# =========================================================
# PREPARE CUSTOMER DATA
# =========================================================

def prepare_customer_data(customer_data):

    if isinstance(customer_data, dict):

        data = pd.DataFrame([customer_data])

    else:

        data = customer_data.copy()

    # Remove spaces from column names
    data.columns = data.columns.str.strip()

    # Convert numeric columns
    numeric_columns = [
        "SeniorCitizen",
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


# =========================================================
# PREDICT CHURN
# =========================================================

def predict_churn(customer_data):

    data = prepare_customer_data(
        customer_data
    )

    model = load_churn_model()
    scaler = load_scaler()

    # -----------------------------------------------------
    # Keep only numeric values for scaler
    # -----------------------------------------------------

    numeric_data = data.select_dtypes(
        include=["number"]
    )

    if numeric_data.empty:

        raise ValueError(
            "No numeric features found in customer data."
        )

    # Fill missing numeric values
    numeric_data = numeric_data.fillna(0)

    # Scale data
    scaled_data = scaler.transform(
        numeric_data
    )

    # ANN prediction
    probability = model.predict(
        scaled_data,
        verbose=0
    )

    # Convert prediction to single number
    churn_probability = float(
        np.asarray(probability).reshape(-1)[0]
    )

    # If model returns percentage
    if churn_probability > 1:
        churn_probability = (
            churn_probability / 100
        )

    # Keep probability between 0 and 1
    churn_probability = max(
        0,
        min(1, churn_probability)
    )

    # -----------------------------------------------------
    # Risk level
    # -----------------------------------------------------

    if churn_probability < 0.30:

        risk_level = "Low"

    elif churn_probability < 0.60:

        risk_level = "Medium"

    elif churn_probability < 0.80:

        risk_level = "High"

    else:

        risk_level = "Critical"

    # -----------------------------------------------------
    # Final prediction
    # -----------------------------------------------------

    if churn_probability >= 0.50:

        prediction = "Likely to Churn"

    else:

        prediction = "Likely to Stay"

    return {
        "churn_probability": round(
            churn_probability * 100,
            2
        ),
        "risk_level": risk_level,
        "prediction": prediction
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("================================")
    print("TELCO CUSTOMER CHURN PREDICTION")
    print("================================")

    try:

        customer = {
            "SeniorCitizen": 0,
            "tenure": 5,
            "MonthlyCharges": 85.50,
            "TotalCharges": 427.50
        }

        result = predict_churn(
            customer
        )

        print(
            "\nChurn Probability:",
            result["churn_probability"],
            "%"
        )

        print(
            "Risk Level:",
            result["risk_level"]
        )

        print(
            "Prediction:",
            result["prediction"]
        )

    except Exception as error:

        print(
            "\nPrediction Error:",
            error
        )
