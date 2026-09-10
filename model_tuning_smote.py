import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def run_tuned_pipeline(data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    print("[1/6] Loading Data & Preprocessing...")
    df = pd.read_csv(data_path)

    # Clean CustomerID & TotalCharges
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    print("[2/6] Train-Test Split & Scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print("[3/6] Applying SMOTE for Class Imbalance...")
    print(f"  - Pre-SMOTE Churn Distribution:\n{y_train.value_counts()}")
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"  - Post-SMOTE Churn Distribution:\n{y_train_res.value_counts()}")

    print("[4/6] Building Tuned ANN with Dropout Layers...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_res.shape[1],)),
        Dropout(0.3),  # Prevents overfitting on oversampled minority class
        Dense(32, activation='relu'),
        Dropout(0.2),  # Prevents overfitting
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )

    # Early stopping to prevent overtraining
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    print("[5/6] Training ANN Model...")
    history = model.fit(
        X_train_res, y_train_res,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    print("[6/6] Evaluating & Saving Visualizations...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Print Detailed Classification Report
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT (SMOTE + TUNED ANN):")
    print("="*50)
    print(classification_report(y_test, y_pred))

    # Plot & Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix (Tuned ANN + SMOTE)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print(" Saved confusion_matrix.png")

    # Plot & Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_val = roc_auc_score(y_test, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()
    print(" Saved roc_curve.png")

if __name__ == '__main__':
    run_tuned_pipeline()
