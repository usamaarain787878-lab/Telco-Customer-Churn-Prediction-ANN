import pandas as pd
import numpy as np

# Sample Telco Customer Churn dataset generate karna
np.random.seed(42)
n_samples = 1000

data = {
    'CustomerID': [f'ID_{i}' for i in range(n_samples)],
    'tenure': np.random.randint(1, 72, size=n_samples),
    'MonthlyCharges': np.random.uniform(20.0, 120.0, size=n_samples),
    'TotalCharges': np.random.uniform(20.0, 8000.0, size=n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples),
    'TechSupport': np.random.choice(['Yes', 'No'], size=n_samples),
    'SeniorCitizen': np.random.choice([0, 1], size=n_samples),
    'Churn': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
}

df = pd.DataFrame(data)
df.to_csv('Cleaned_Dataset.csv', index=False)
print("Cleaned_Dataset.csv successfully generated!")
