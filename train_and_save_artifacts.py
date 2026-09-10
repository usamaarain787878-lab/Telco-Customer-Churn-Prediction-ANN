import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def run_pipeline(data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    print("[1/5] Loading and Cleaning Data...")
    df = pd.read_csv(data_path)
    
    # Drop customerID as it's not useful for prediction
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric, coerce errors to NaN and fill with median
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Convert Churn target column to binary (0/1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print("[2/5] Encoding Categorical Variables...")
    # Separate categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # One-Hot Encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    # Save feature names list to ensure consistent input during Streamlit deployment
    feature_names = X.columns.tolist()
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    print(" Saved feature_names.pkl")

    print("[3/5] Train-Test Split & Scaling...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale Numerical Features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Save Scaler
    joblib.dump(scaler, 'scaler.pkl')
    print(" Saved scaler.pkl")

    print("[4/5] Applying SMOTE for Class Imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"  - Training samples before SMOTE: {len(X_train)}")
    print(f"  - Training samples after SMOTE: {len(X_train_resampled)}")

    print("[5/5] Building and Training ANN Model...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Save Trained ANN Model
    model.save('churn_model.h5')
    print(" Saved churn_model.h5")

    # Evaluate Model
    loss, accuracy, recall, auc = model.evaluate(X_test, y_test, verbose=0)
    print("\n" + "="*40)
    print("MODEL PERFORMANCE ON TEST DATA:")
    print(f"  - Accuracy : {accuracy*100:.2f}%")
    print(f"  - Recall   : {recall*100:.2f}%")
    print(f"  - AUC Score: {auc:.4f}")
    print("="*40)
    print("\nAll artifacts saved successfully! You are ready to build app.py")

if __name__ == '__main__':
    run_pipeline()
