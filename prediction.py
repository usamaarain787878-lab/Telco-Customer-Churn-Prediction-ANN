import os
import joblib
import pandas as pd
import tensorflow as tf

def load_artifacts():
    # Model check karein (har possible path aur extension par)
    model_paths = [
        "churn_model.keras",
        "churn_model.h5",
        "models/churn_model.keras",
        "models/churn_model.h5"
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            break

    # Scaler check karein
    scaler_paths = ["scaler.pkl", "models/scaler.pkl"]
    scaler = None
    for path in scaler_paths:
        if os.path.exists(path):
            scaler = joblib.load(path)
            break

    # Feature names check karein
    feature_paths = ["feature_names.pkl", "models/feature_names.pkl"]
    feature_names = None
    for path in feature_paths:
        if os.path.exists(path):
            feature_names = joblib.load(path)
            break

    return model, scaler, feature_names

def predict_churn(customer_data):
    """
    Accepts dictionary of input features and returns prediction details.
    """
    model, scaler, feature_names = load_artifacts()

    if model is None or scaler is None or feature_names is None:
        raise FileNotFoundError("Model or preprocessing artifacts (.h5/.keras/.pkl) missing!")

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([customer_data])

    # One-Hot Encoding matching features
    input_encoded = pd.get_dummies(input_df)

    # Align columns with model features
    full_df = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        full_df[col] = input_encoded[col] if col in input_encoded.columns else 0

    # Fill NaNs with 0
    full_df = full_df.fillna(0)

    # Scale numeric columns
    scaled_input = scaler.transform(full_df)

    # Predict using model
    prob = float(model.predict(scaled_input, verbose=0)[0][0]) * 100

    if prob >= 70:
        risk_level = "High / Critical"
        prediction = "Churn Risk"
    elif prob >= 40:
        risk_level = "Medium Risk"
        prediction = "Watchlist"
    else:
        risk_level = "Low Risk"
        prediction = "Retained"

    return {
        "churn_probability": round(prob, 2),
        "risk_level": risk_level,
        "prediction": prediction
    }
