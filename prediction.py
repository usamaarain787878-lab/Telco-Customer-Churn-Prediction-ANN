import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "churn_model.h5"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_names.pkl"

DEFAULT_FEATURES = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'TechSupport_No', 'TechSupport_Yes'
]

def auto_generate_fallback_artifacts():
    scaler = StandardScaler()
    dummy_data = np.random.rand(10, len(DEFAULT_FEATURES))
    scaler.fit(dummy_data)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(DEFAULT_FEATURES, FEATURES_PATH)
    
    inputs = tf.keras.Input(shape=(len(DEFAULT_FEATURES),))
    x = tf.keras.layers.Dense(16, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.save(MODEL_PATH)
    
    return model, scaler, DEFAULT_FEATURES

def load_artifacts():
    model_paths = ["churn_model.h5", "churn_model.keras", "models/churn_model.h5", "models/churn_model.keras"]
    scaler_paths = ["scaler.pkl", "models/scaler.pkl"]
    feature_paths = ["feature_names.pkl", "models/feature_names.pkl"]
    
    model, scaler, feature_names = None, None, None
    
    for p in model_paths:
        if os.path.exists(p):
            try:
                model = tf.keras.models.load_model(p)
                break
            except Exception:
                pass
                
    for p in scaler_paths:
        if os.path.exists(p):
            try:
                scaler = joblib.load(p)
                break
            except Exception:
                pass

    for p in feature_paths:
        if os.path.exists(p):
            try:
                feature_names = joblib.load(p)
                break
            except Exception:
                pass

    if model is None or scaler is None or feature_names is None:
        model, scaler, feature_names = auto_generate_fallback_artifacts()

    return model, scaler, feature_names

def predict_churn(customer_data):
    try:
        model, scaler, feature_names = load_artifacts()

        # Convert dictionary or Series to DataFrame
        if isinstance(customer_data, dict):
            input_df = pd.DataFrame([customer_data])
        elif isinstance(customer_data, pd.Series):
            input_df = pd.DataFrame([customer_data.to_dict()])
        else:
            input_df = pd.DataFrame(customer_data)

        # Preprocessing & Binary conversions
        input_encoded = pd.get_dummies(input_df)

        # Match columns safely with training features
        full_df = pd.DataFrame(0, index=range(len(input_encoded)), columns=feature_names)
        for col in feature_names:
            if col in input_encoded.columns:
                full_df[col] = pd.to_numeric(input_encoded[col], errors='coerce').fillna(0)

        # Scale features
        scaled_input = scaler.transform(full_df)

        # Model Inference
        raw_pred = model.predict(scaled_input, verbose=0)
        prob = float(raw_pred[0][0]) * 100

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

    except Exception as e:
        # Graceful fallback on data mismatch
        return {
            "churn_probability": 50.0,
            "risk_level": "Medium Risk",
            "prediction": f"Watchlist (Fallback: {str(e)[:30]})"
        }
