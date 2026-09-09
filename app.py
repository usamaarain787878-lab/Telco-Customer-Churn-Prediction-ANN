import streamlit as st
import pandas as pd
import plotly.express as px

# Page Setup
st.set_page_config(page_title="Telco Churn & Transaction Analytics", layout="wide")

# Header Section
st.title("📊 Industrial Data Cleaning & Sales Analytics Dashboard")
st.markdown("### **Project Purpose & Overview**")
st.info("""
**Business Objective:** Unstructured raw transactional data ko automated data pipeline ke zariye clean, normalize, aur analyze karna. 
Yeh project data anomaly detection, missing value handling, aur city-wise sales performance visualization ke liye banaya gaya hai.
""")

# Key Features / Highlights
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("#### 🧹 **Data Cleaning**")
    st.write("Null values removal, data type normalization, and timestamp parsing.")
with col_b:
    st.markdown("#### 📈 **EDA & Metrics**")
    st.write("Summary statistics, city-wise aggregation, and transaction tracking.")
with col_c:
    st.markdown("#### 🚀 **Production Ready**")
    st.write("Deployable data pipeline for real-time analytics integration.")

st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_Dataset.csv")

try:
    df = load_data()

    # Executive KPI Cards
    st.subheader("📌 Key Executive Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Transactions", len(df))
    kpi2.metric("Total Revenue", f"${df['Value'].sum():,.2f}")
    kpi3.metric("Avg Transaction Value", f"${df['Value'].mean():,.2f}")
    kpi4.metric("Unique Products", df['Product'].nunique())

    st.markdown("---")

    # Visualizations Section
    st.subheader("📊 Interactive Business Insights")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Revenue Distribution by City**")
        fig_city = px.bar(df, x='City', y='Value', color='Product', barmode='group',
                          title="City vs Total Order Value")
        st.plotly_chart(fig_city, use_container_width=True)

    with chart_col2:
        st.markdown("**Order Quantity Share by Product**")
        fig_prod = px.pie(df, names='Product', values='Qty', title="Product Quantity Breakdown")
        st.plotly_chart(fig_prod, use_container_width=True)

    st.markdown("---")

    # Data Tables Section
    tab1, tab2 = st.tabs(["📋 Cleaned Dataset", "📐 Statistical Summary"])
    
    with tab1:
        st.dataframe(df, use_container_width=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

except Exception as e:
    st.error(f"Error loading dataset: {e}")
