# 📡 AI Telco Customer Churn Prediction & Retention System (ANN)

An end-to-end Machine Learning & Deep Learning enterprise solution designed to predict customer churn in the telecommunications industry, quantify financial risk exposure, and offer automated AI-driven retention strategies.

---

## 🎯 Executive Summary & Business Problem
Customer retention is critical for telecom service providers. Acquiring new customers costs 5x more than retaining existing ones. This project leverages an **Artificial Neural Network (ANN)** trained on telecom telemetry data to identify high-risk accounts prior to churn.

### 🌟 Key Features
- **Deep Learning Classification:** Custom Artificial Neural Network built using TensorFlow/Keras.
- **Class Imbalance Resolution:** Applied **SMOTE** (Synthetic Minority Over-sampling) to improve minority class recall.
- **Interactive Executive Dashboard:** Built with **Streamlit** for real-time customer churn probability evaluation.
- **Financial Risk Calculator:** Real-time revenue exposure analysis for high-risk customer segments.
- **Explainable AI (XAI):** Decision attribution breakdown mapping churn factors directly to retention recommendations.

---

## 🏗️ Technical Architecture & Pipeline

1. **Data Preprocessing & Scaling:**
   - Handled missing values (`TotalCharges`).
   - Binary & One-Hot Encoding for categorical features.
   - Feature scaling using `StandardScaler`.
2. **Imbalance Handling:**
   - Over-sampling using `SMOTE` to address 73/27 class imbalance ratio.
3. **ANN Architecture:**
   - Input Layer (Dense + ReLU)
   - Hidden Layer 1 (Dense + ReLU + Dropout 0.2)
   - Hidden Layer 2 (Dense + ReLU + Dropout 0.2)
   - Output Layer (Sigmoid Activation for Binary Classification)
4. **Model Optimization:**
   - Loss Function: `binary_crossentropy`
   - Optimizer: `Adam`
   - Callbacks: `EarlyStopping` (patience=10, restore_best_weights=True)

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~82% - 85% |
| **Precision** | ~80% |
| **Recall (Churn Class)** | ~81% (Enhanced via SMOTE) |
| **F1-Score** | ~80% |

---

## 📁 Repository Structure

```text
├── app.py                         # Streamlit Interactive Web Application UI
├── prediction.py                  # Real-time inference wrapper script
├── train_and_save_artifacts.py    # Training & Artifact Export Pipeline
├── model_tuning_smote.py          # Class imbalance handling & ANN tuning script
├── Cleaned_Dataset.csv            # Preprocessed Telco Customer Dataset
├── churn_model.h5                 # Trained Keras ANN Model Weights
├── scaler.pkl                     # Saved StandardScaler instance
├── feature_names.pkl              # Saved Feature Column Names
├── requirements.txt               # Python Dependencies
└── README.md                      # Project Documentation
