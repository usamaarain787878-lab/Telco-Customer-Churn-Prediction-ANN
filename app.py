import streamlit as st
import pandas as pd

st.set_page_config(page_title="Telco Churn Dataset Analysis", layout="wide")

st.title("📊 Telco Customer Churn - Data Analysis")
st.write("Welcome to the Telco Customer Churn project dashboard.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_Dataset.csv")

try:
    df = load_data()
    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("Dataset Summary Statistics")
    st.write(df.describe())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
