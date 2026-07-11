import pandas as pd
import numpy as np
import os
import sys

print("\n==========================================================================")
print("🚀 DECODELABS INDUSTRIAL INTERNSHIP - DATA INTEGRITY PIPELINE")
print("==========================================================================")

file_name = "clean_gold_standard_dataset.csv"

# Agar file nahi hai toh automatic dummy data create karna
if not os.path.exists(file_name):
    print(f"[System Log] '{file_name}' nahi mili. System automatic generating data...")
    dummy_data = {
        'Order_ID': ['101', '102', '103', '101', '104'],
        'Product': ['Laptop', 'Mouse', ' Keyboard', 'Laptop', 'Monitor'],
        'Qty': [1.0, np.nan, 3.0, 1.0, 2.0],
        'Value': [1200.506, 25.0, np.nan, 1200.506, 350.00],
        'City': ['hyderabad ', 'karachi', 'Lahore', 'hyderabad ', ' islamabad'],
        'Timestamp': ['2026-07-01 10:00:00', '2026-07-01 11:30:00', np.nan, '2026-07-01 10:00:00', '2026-07-01 14:15:00']
    }
    df_raw = pd.DataFrame(dummy_data)
    df_raw.to_csv(file_name, index=False)
    print(f"✅ Raw test file successfully created inside: {os.getcwd()}")

try:
    # Ab file load hogi safely
    df = pd.read_csv(file_name)
    print(f"✅ Success: Dataset Ingested. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # 1. Clean Column Names
    df.columns = df.columns.str.strip()

    # 2. Strategic Imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # === YEH LINE CHANGE KI HAI TAAKE WARNING NA AAYE ===
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Formatting Standardization
    if 'City' in df.columns:
        df['City'] = df['City'].astype(str).str.strip().str.title()

    float_cols = df.select_dtypes(include=['float64', 'float32']).columns
    for col in float_cols:
        df[col] = df[col].round(2)

    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['Timestamp'] = df['Timestamp'].ffill()

    # 4. Duplicates Management
    initial_dups = df.duplicated().sum()
    if 'Order_ID' in df.columns:
        df['Order_ID'] = df['Order_ID'].astype(str).str.strip()
        df.drop_duplicates(subset=['Order_ID'], keep='first', inplace=True)
    df.drop_duplicates(inplace=True)

    print("\n==================== QUALITY MATRIX REPORT ====================")
    print(f"  1. Initial Transformed Duplicates Cleaned: {initial_dups}")
    print(f"  2. Total Inflated Record Duplicates Found : {df.duplicated().sum()}")
    print(f"  3. Total Missing/Null Gaps Remaining      : {df.isnull().sum().sum()}")
    print("===============================================================")

    # Export clean final file
    output_file = "Cleaned_Dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\n🎉 PIPELINE SUCCESS: '{output_file}' has been generated successfully!")
    print("==========================================================================\n")

except Exception as e:
    print(f"\n❌ Pipeline Runtime Error: {str(e)}")