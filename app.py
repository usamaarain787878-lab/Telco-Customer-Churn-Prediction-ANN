import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Telco Customer Churn Prediction",
    page_icon="📡",
    layout="wide"
)


# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666666;
}

.info-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f5f7fa;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📡 AI Telco Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Artificial Intelligence powered customer churn analysis and prediction platform'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# PROJECT PURPOSE
# =========================================================

st.header("🎯 Project Purpose")

st.info("""
### Business Objective

Telecommunication companies can lose significant revenue when customers
leave their services.

This project uses **Artificial Intelligence and Machine Learning** to analyze
customer information and predict whether a customer is likely to churn.

The system is designed to help businesses:

• Identify customers who are at risk of leaving  
• Understand customer churn patterns  
• Analyze important customer characteristics  
• Estimate the level of churn risk  
• Support customer retention strategies  
""")


# =========================================================
# HOW THE SYSTEM WORKS
# =========================================================

st.header("⚙️ How This AI System Works")

step1, step2, step3, step4 = st.columns(4)

with step1:
    st.markdown("### 1️⃣")
    st.markdown("**Customer Data**")
    st.write(
        "Customer demographics, services, contract and billing information "
        "are provided to the system."
    )

with step2:
    st.markdown("### 2️⃣")
    st.markdown("**Data Processing**")
    st.write(
        "The data is cleaned, transformed and prepared for the AI model."
    )

with step3:
    st.markdown("### 3️⃣")
    st.markdown("**ANN Prediction**")
    st.write(
        "An Artificial Neural Network analyzes customer characteristics "
        "and calculates churn probability."
    )

with step4:
    st.markdown("### 4️⃣")
    st.markdown("**Business Action**")
    st.write(
        "High-risk customers can be targeted with appropriate retention "
        "strategies."
    )


st.markdown("---")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv("Cleaned_Dataset.csv")


try:

    df = load_data()

except Exception as e:

    st.error("❌ Dataset could not be loaded.")

    st.write("Please make sure this file exists:")

    st.code("Cleaned_Dataset.csv")

    st.stop()


# =========================================================
# BASIC DATA CLEANING
# =========================================================

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()


# =========================================================
# EXECUTIVE METRICS
# =========================================================

st.header("📊 Executive Dashboard")

total_customers = len(df)


# Detect churn column
churn_column = None

possible_churn_columns = [
    "Churn",
    "churn",
    "Exited",
    "Customer_Status"
]

for column in possible_churn_columns:

    if column in df.columns:
        churn_column = column
        break


# Calculate churn customers
if churn_column:

    churn_values = df[churn_column].astype(str).str.lower()

    churned_customers = churn_values.isin(
        ["yes", "true", "1", "churned", "churn"]
    ).sum()

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers > 0 else 0
    )

else:

    churned_customers = 0
    churn_rate = 0


# Detect monthly charges
monthly_charge_column = None

possible_charge_columns = [
    "MonthlyCharges",
    "Monthly Charges",
    "MonthlyCharge"
]

for column in possible_charge_columns:

    if column in df.columns:
        monthly_charge_column = column
        break


if monthly_charge_column:

    df[monthly_charge_column] = pd.to_numeric(
        df[monthly_charge_column],
        errors="coerce"
    )

    average_monthly_charge = df[
        monthly_charge_column
    ].mean()

else:

    average_monthly_charge = 0


# =========================================================
# KPI CARDS
# =========================================================

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.metric(
        "👥 Total Customers",
        f"{total_customers:,}"
    )


with kpi2:

    st.metric(
        "🚪 Churned Customers",
        f"{churned_customers:,}"
    )


with kpi3:

    st.metric(
        "📉 Churn Rate",
        f"{churn_rate:.2f}%"
    )


with kpi4:

    st.metric(
        "💳 Avg Monthly Charges",
        f"${average_monthly_charge:,.2f}"
    )


st.markdown("---")


# =========================================================
# CHURN ANALYSIS
# =========================================================

if churn_column:

    st.header("📈 Customer Churn Analysis")

    chart1, chart2 = st.columns(2)


    # -----------------------------------------------------
    # Churn Distribution
    # -----------------------------------------------------

    with chart1:

        churn_count = df[churn_column].astype(str).value_counts()

        fig_churn = px.pie(
            values=churn_count.values,
            names=churn_count.index,
            title="Customer Churn Distribution",
            hole=0.4
        )

        st.plotly_chart(
            fig_churn,
            use_container_width=True
        )


    # -----------------------------------------------------
    # Churn Bar Chart
    # -----------------------------------------------------

    with chart2:

        fig_bar = px.bar(
            x=churn_count.index,
            y=churn_count.values,
            title="Churned vs Non-Churned Customers",
            labels={
                "x": "Customer Status",
                "y": "Number of Customers"
            }
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )


else:

    st.warning(
        "⚠️ Churn column was not found in the dataset."
    )


# =========================================================
# CONTRACT ANALYSIS
# =========================================================

if "Contract" in df.columns and churn_column:

    st.header("📄 Churn Analysis by Contract")

    contract_churn = pd.crosstab(
        df["Contract"],
        df[churn_column]
    )

    fig_contract = px.bar(
        contract_churn,
        barmode="group",
        title="Customer Churn by Contract Type"
    )

    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )


# =========================================================
# INTERNET SERVICE ANALYSIS
# =========================================================

if "InternetService" in df.columns and churn_column:

    st.header("🌐 Churn Analysis by Internet Service")

    internet_churn = pd.crosstab(
        df["InternetService"],
        df[churn_column]
    )

    fig_internet = px.bar(
        internet_churn,
        barmode="group",
        title="Customer Churn by Internet Service"
    )

    st.plotly_chart(
        fig_internet,
        use_container_width=True
    )


# =========================================================
# CUSTOMER TENURE ANALYSIS
# =========================================================

if "tenure" in df.columns and churn_column:

    st.header("📅 Customer Tenure Analysis")

    tenure_data = df.copy()

    tenure_data["tenure"] = pd.to_numeric(
        tenure_data["tenure"],
        errors="coerce"
    )

    fig_tenure = px.histogram(
        tenure_data,
        x="tenure",
        color=churn_column,
        nbins=20,
        title="Customer Churn by Tenure"
    )

    st.plotly_chart(
        fig_tenure,
        use_container_width=True
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.header("💡 Business Insights")

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("### 🎯 Customer Retention")

    st.write("""
    Identifying customers with high churn probability allows the business
    to take preventive action before the customer leaves.
    """)


with col2:

    st.markdown("### 💰 Revenue Protection")

    st.write("""
    Reducing customer churn can help telecom companies protect recurring
    monthly revenue and improve customer lifetime value.
    """)


with col3:

    st.markdown("### 🤖 AI Decision Support")

    st.write("""
    The ANN model provides a data-driven approach for identifying customers
    who may require additional attention.
    """)


# =========================================================
# DATASET INFORMATION
# =========================================================

st.markdown("---")

st.header("📋 Dataset Information")

tab1, tab2 = st.tabs(
    [
        "👥 Customer Data",
        "📐 Statistical Summary"
    ]
)


with tab1:

    st.dataframe(
        df,
        use_container_width=True
    )


with tab2:

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True
    )


# =========================================================
# TECHNOLOGY STACK
# =========================================================

st.markdown("---")

st.header("🛠️ Technology Stack")

tech1, tech2, tech3, tech4, tech5 = st.columns(5)

with tech1:
    st.markdown("🐍 **Python**")

with tech2:
    st.markdown("🐼 **Pandas**")

with tech3:
    st.markdown("📊 **Plotly**")

with tech4:
    st.markdown("🤖 **ANN / Deep Learning**")

with tech5:
    st.markdown("🚀 **Streamlit**")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Telco Customer Churn Prediction | "
    "Customer Analytics & Retention Intelligence Platform"
)
