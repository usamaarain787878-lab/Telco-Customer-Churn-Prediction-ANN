```python
import streamlit as st
import pandas as pd
import plotly.express as px

# Step 10 prediction module
from prediction import predict_churn

# Step 12 prediction history
from prediction_history import save_prediction, load_prediction_history

# Step 13 recommendations
from recommendations import get_recommendations


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
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Executive Dashboard",
        "🔮 Churn Prediction",
        "📊 Customer Analytics",
        "📜 Prediction History",
        "💡 Retention Recommendations",
        "ℹ️ About Project"
    ]
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv("Cleaned_Dataset.csv")


try:

    df = load_data()

except Exception:

    st.error("❌ Dataset could not be loaded.")

    st.write(
        "Please make sure this file exists:"
    )

    st.code("Cleaned_Dataset.csv")

    st.stop()


# Remove spaces from column names
df.columns = df.columns.str.strip()


# =========================================================
# PAGE 1 — EXECUTIVE DASHBOARD
# =========================================================

if page == "🏠 Executive Dashboard":

    st.header("🎯 Project Purpose")

    st.info("""
    ### Business Objective

    Telecommunication companies can lose significant revenue when customers
    leave their services.

    This platform uses Artificial Intelligence and Machine Learning to analyze
    customer information and predict whether a customer is likely to churn.

    The system helps businesses:

    • Identify customers at risk of leaving
    • Understand customer churn patterns
    • Analyze customer characteristics
    • Estimate churn risk
    • Support customer retention strategies
    """)


    # -----------------------------------------------------
    # HOW SYSTEM WORKS
    # -----------------------------------------------------

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
            "The Artificial Neural Network analyzes customer characteristics "
            "and calculates churn probability."
        )

    with step4:

        st.markdown("### 4️⃣")
        st.markdown("**Business Action**")

        st.write(
            "High-risk customers can be targeted with retention strategies."
        )


    st.markdown("---")


    # -----------------------------------------------------
    # EXECUTIVE METRICS
    # -----------------------------------------------------

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


    # Calculate churn

    if churn_column:

        churn_values = (
            df[churn_column]
            .astype(str)
            .str.lower()
        )

        churned_customers = churn_values.isin(
            [
                "yes",
                "true",
                "1",
                "churned",
                "churn"
            ]
        ).sum()

        churn_rate = (
            churned_customers / total_customers * 100
            if total_customers > 0
            else 0
        )

    else:

        churned_customers = 0
        churn_rate = 0


    # Monthly charges

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


    # KPI cards

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


    # -----------------------------------------------------
    # CHURN ANALYSIS
    # -----------------------------------------------------

    if churn_column:

        st.header("📈 Customer Churn Analysis")

        chart1, chart2 = st.columns(2)

        with chart1:

            churn_count = (
                df[churn_column]
                .astype(str)
                .value_counts()
            )

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


    # -----------------------------------------------------
    # CONTRACT ANALYSIS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # INTERNET SERVICE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TENURE ANALYSIS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

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
# PAGE 2 — CHURN PREDICTION
# =========================================================

elif page == "🔮 Churn Prediction":

    st.header("🔮 AI Customer Churn Prediction")

    st.write(
        "Enter customer information below and use the trained ANN model "
        "to estimate churn probability."
    )

    st.markdown("---")


    # -----------------------------------------------------
    # CUSTOMER ID
    # -----------------------------------------------------

    customer_id = st.text_input(
        "Customer ID",
        value="CUST001"
    )


    # -----------------------------------------------------
    # NUMERIC FEATURES
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

    with col2:

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

    with col3:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )


    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


    # -----------------------------------------------------
    # CUSTOMER SERVICES
    # -----------------------------------------------------

    col4, col5 = st.columns(2)

    with col4:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col5:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No"
            ]
        )


    st.markdown("---")


    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button(
        "🤖 Predict Customer Churn",
        use_container_width=True
    ):

        try:

            customer_data = {
                "SeniorCitizen": senior_citizen,
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }


            # Actual ANN prediction

            result = predict_churn(
                customer_data
            )


            probability = result[
                "churn_probability"
            ]

            risk_level = result[
                "risk_level"
            ]

            prediction = result[
                "prediction"
            ]


            st.markdown("---")

            st.subheader("🎯 Prediction Result")


            result1, result2, result3 = st.columns(3)

            with result1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.2f}%"
                )

            with result2:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            with result3:

                st.metric(
                    "Prediction",
                    prediction
                )


            # -------------------------------------------------
            # SAVE HISTORY
            # -------------------------------------------------

            save_prediction(
                customer_id=customer_id,
                churn_probability=probability,
                risk_level=risk_level,
                prediction=prediction
            )

            st.success(
                "✅ Prediction completed and saved to prediction history."
            )


            # -------------------------------------------------
            # RECOMMENDATIONS
            # -------------------------------------------------

            st.subheader("💡 Recommended Retention Actions")

            recommendations = get_recommendations(
                contract=contract,
                monthly_charges=monthly_charges,
                tenure=tenure,
                tech_support=tech_support
            )

            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )


        except Exception as error:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.code(
                str(error)
            )


# =========================================================
# PAGE 3 — CUSTOMER ANALYTICS
# =========================================================

elif page == "📊 Customer Analytics":

    st.header("📊 Customer Analytics")

    st.write(
        "Explore customer behavior and churn patterns."
    )


    if "Contract" in df.columns and churn_column:

        contract_data = pd.crosstab(
            df["Contract"],
            df[churn_column]
        )

        fig = px.bar(
            contract_data,
            barmode="group",
            title="Churn by Contract"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    if "InternetService" in df.columns and churn_column:

        internet_data = pd.crosstab(
            df["InternetService"],
            df[churn_column]
        )

        fig = px.bar(
            internet_data,
            barmode="group",
            title="Churn by Internet Service"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    if "PaymentMethod" in df.columns and churn_column:

        payment_data = pd.crosstab(
            df["PaymentMethod"],
            df[churn_column]
        )

        fig = px.bar(
            payment_data,
            barmode="group",
            title="Churn by Payment Method"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# PAGE 4 — PREDICTION HISTORY
# =========================================================

elif page == "📜 Prediction History":

    st.header("📜 Prediction History")

    history = load_prediction_history()


    if history.empty:

        st.info(
            "No prediction history available yet."
        )

    else:

        st.dataframe(
            history,
            use_container_width=True
        )


        # Download history

        csv_data = history.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv"
        )


# =========================================================
# PAGE 5 — RETENTION RECOMMENDATIONS
# =========================================================

elif page == "💡 Retention Recommendations":

    st.header("💡 Customer Retention Recommendations")

    st.write(
        "Generate business actions based on customer characteristics."
    )


    col1, col2 = st.columns(2)

    with col1:

        recommendation_contract = st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            key="recommendation_contract"
        )

        recommendation_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            key="recommendation_charges"
        )


    with col2:

        recommendation_tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=100,
            value=12,
            key="recommendation_tenure"
        )

        recommendation_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No"
            ],
            key="recommendation_support"
        )


    if st.button(
        "💡 Generate Recommendations",
        use_container_width=True
    ):

        recommendations = get_recommendations(
            contract=recommendation_contract,
            monthly_charges=recommendation_charges,
            tenure=recommendation_tenure,
            tech_support=recommendation_support
        )


        st.subheader(
            "Recommended Actions"
        )

        for recommendation in recommendations:

            st.success(
                recommendation
            )


# =========================================================
# PAGE 6 — ABOUT
# =========================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About This Project")

    st.write("""
    ## AI-Powered Telecom Customer Churn Prediction & Retention Platform

    This project uses Artificial Neural Networks (ANN) and customer analytics
    to identify telecom customers who may be at risk of leaving.

    ### Main Capabilities

    • Customer churn prediction

    • Churn probability estimation

    • Customer risk classification

    • Customer analytics

    • Prediction history

    • Retention recommendations

    • Revenue risk analysis

    • Batch prediction

    • Model performance monitoring

    ### Technology Stack

    🐍 Python

    🐼 Pandas

    📊 Plotly

    🤖 TensorFlow / Artificial Neural Network

    🚀 Streamlit

    ### Business Value

    The system is designed to help telecom businesses identify customers
    who may leave and take preventive retention actions before churn occurs.
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Telco Customer Churn Prediction | "
    "Customer Analytics & Retention Intelligence Platform"
)
```
