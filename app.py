import streamlit as st
import pandas as pd
import plotly.express as px

# =========================================================
# PROJECT MODULES
# =========================================================

from prediction import predict_churn

from prediction_history import (
    save_prediction,
    load_prediction_history
)

from recommendations import get_recommendations

from revenue_risk import (
    calculate_revenue_risk,
    calculate_high_risk_revenue
)

from model_performance import (
    calculate_metrics,
    create_confusion_matrix,
    create_metrics_dataframe
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Telco Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
    }

    .risk-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .footer {
        text-align: center;
        color: #777777;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📡 AI Telco Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Artificial Intelligence powered customer churn analysis, prediction '
    'and retention intelligence platform'
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
        "💰 Revenue Risk",
        "📈 Model Performance",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **AI Telco Churn Platform**

    Use the navigation menu to explore:

    • Customer analytics  
    • AI churn prediction  
    • Retention actions  
    • Revenue risk  
    • Prediction history  
    • Model performance
    """
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("Cleaned_Dataset.csv")

    data.columns = data.columns.str.strip()

    return data


try:

    df = load_data()

except Exception as error:

    st.error("❌ Dataset could not be loaded.")

    st.write(
        "Please make sure the following file exists:"
    )

    st.code("Cleaned_Dataset.csv")

    st.stop()


# =========================================================
# COMMON COLUMN DETECTION
# =========================================================

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


# =========================================================
# PAGE 1
# EXECUTIVE DASHBOARD
# =========================================================

if page == "🏠 Executive Dashboard":

    st.header("🎯 Executive Dashboard")

    st.info(
        """
        ### Business Objective

        Telecommunication companies can lose significant revenue when
        customers leave their services.

        This platform uses Artificial Intelligence, customer analytics and
        predictive modeling to identify customers who may be at risk of churn.

        The system helps businesses:

        • Identify customers at risk of leaving
        • Understand churn patterns
        • Analyze customer behavior
        • Estimate potential revenue risk
        • Support customer retention strategies
        """
    )

    # -----------------------------------------------------
    # HOW SYSTEM WORKS
    # -----------------------------------------------------

    st.header("⚙️ How This AI System Works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:

        st.markdown("### 1️⃣")
        st.markdown("**Customer Data**")

        st.write(
            "Customer demographics, services, contract and billing "
            "information are provided to the system."
        )

    with step2:

        st.markdown("### 2️⃣")
        st.markdown("**Data Processing**")

        st.write(
            "Customer data is cleaned, transformed and prepared "
            "for machine learning."
        )

    with step3:

        st.markdown("### 3️⃣")
        st.markdown("**ANN Prediction**")

        st.write(
            "The Artificial Neural Network calculates the customer's "
            "churn probability."
        )

    with step4:

        st.markdown("### 4️⃣")
        st.markdown("**Business Action**")

        st.write(
            "High-risk customers can receive targeted retention actions."
        )

    st.markdown("---")

    # -----------------------------------------------------
    # KPI CALCULATIONS
    # -----------------------------------------------------

    total_customers = len(df)

    if churn_column:

        churn_values = (
            df[churn_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        churned_customers = churn_values.isin(
            [
                "yes",
                "true",
                "1",
                "churn",
                "churned"
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

    if monthly_charge_column:

        charges = pd.to_numeric(
            df[monthly_charge_column],
            errors="coerce"
        )

        average_monthly_charge = charges.mean()

    else:

        average_monthly_charge = 0

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    st.header("📊 Business Overview")

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
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    if churn_column:

        st.header("📈 Customer Churn Analysis")

        chart1, chart2 = st.columns(2)

        churn_count = (
            df[churn_column]
            .astype(str)
            .value_counts()
        )

        with chart1:

            fig_churn = px.pie(
                values=churn_count.values,
                names=churn_count.index,
                title="Customer Churn Distribution",
                hole=0.45
            )

            st.plotly_chart(
                fig_churn,
                use_container_width=True
            )

        with chart2:

            fig_bar = px.bar(
                x=churn_count.index,
                y=churn_count.values,
                title="Customer Churn Count",
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

        st.header("📄 Churn by Contract")

        contract_data = pd.crosstab(
            df["Contract"],
            df[churn_column]
        )

        fig_contract = px.bar(
            contract_data,
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

        st.header("🌐 Churn by Internet Service")

        internet_data = pd.crosstab(
            df["InternetService"],
            df[churn_column]
        )

        fig_internet = px.bar(
            internet_data,
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

        try:

            st.dataframe(
                df.describe(include="all"),
                use_container_width=True
            )

        except Exception:

            st.dataframe(
                df.describe(),
                use_container_width=True
            )


# =========================================================
# PAGE 2
# CHURN PREDICTION
# =========================================================

elif page == "🔮 Churn Prediction":

    st.header("🔮 AI Customer Churn Prediction")

    st.write(
        "Enter customer information and use the trained ANN model "
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
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "🤖 Predict Customer Churn",
        use_container_width=True
    ):

        if not customer_id.strip():

            st.warning(
                "Please enter a Customer ID."
            )

            st.stop()

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
            # PROBABILITY BAR
            # -------------------------------------------------

            st.subheader("📊 Churn Probability")

            st.progress(
                min(
                    max(
                        int(probability),
                        0
                    ),
                    100
                )
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
            # RETENTION RECOMMENDATIONS
            # -------------------------------------------------

            st.subheader(
                "💡 Recommended Retention Actions"
            )

            recommendations = get_recommendations(
                contract=contract,
                monthly_charges=monthly_charges,
                tenure=tenure,
                tech_support=tech_support
            )

            for recommendation in recommendations:

                st.info(
                    f"💡 {recommendation}"
                )

        except Exception as error:

            st.error(
                "❌ Prediction could not be completed."
            )

            st.warning(
                "Please verify that the ANN model and scaler "
                "match the features expected by prediction.py."
            )

            st.code(
                str(error)
            )


# =========================================================
# PAGE 3
# CUSTOMER ANALYTICS
# =========================================================

elif page == "📊 Customer Analytics":

    st.header("📊 Customer Analytics")

    st.write(
        "Explore customer behavior and identify important churn patterns."
    )

    st.markdown("---")

    if "Contract" in df.columns and churn_column:

        st.subheader("📄 Contract Analysis")

        contract_data = pd.crosstab(
            df["Contract"],
            df[churn_column]
        )

        fig = px.bar(
            contract_data,
            barmode="group",
            title="Customer Churn by Contract"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "InternetService" in df.columns and churn_column:

        st.subheader("🌐 Internet Service Analysis")

        internet_data = pd.crosstab(
            df["InternetService"],
            df[churn_column]
        )

        fig = px.bar(
            internet_data,
            barmode="group",
            title="Customer Churn by Internet Service"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "PaymentMethod" in df.columns and churn_column:

        st.subheader("💳 Payment Method Analysis")

        payment_data = pd.crosstab(
            df["PaymentMethod"],
            df[churn_column]
        )

        fig = px.bar(
            payment_data,
            barmode="group",
            title="Customer Churn by Payment Method"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if (
        "MonthlyCharges" in df.columns
        and churn_column
    ):

        st.subheader("💰 Monthly Charges vs Churn")

        charge_data = df.copy()

        charge_data["MonthlyCharges"] = pd.to_numeric(
            charge_data["MonthlyCharges"],
            errors="coerce"
        )

        charge_data = charge_data.dropna(
            subset=["MonthlyCharges"]
        )

        fig = px.box(
            charge_data,
            x=churn_column,
            y="MonthlyCharges",
            title="Monthly Charges by Churn Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# PAGE 4
# PREDICTION HISTORY
# =========================================================

elif page == "📜 Prediction History":

    st.header("📜 Prediction History")

    st.write(
        "Review previously generated customer churn predictions."
    )

    st.markdown("---")

    history = load_prediction_history()

    if history.empty:

        st.info(
            "No prediction history available yet."
        )

    else:

        # -------------------------------------------------
        # HISTORY KPIs
        # -------------------------------------------------

        total_predictions = len(history)

        critical_predictions = (
            history["Risk_Level"]
            .astype(str)
            .str.lower()
            .eq("critical")
            .sum()
        )

        high_predictions = (
            history["Risk_Level"]
            .astype(str)
            .str.lower()
            .eq("high")
            .sum()
        )

        h1, h2, h3 = st.columns(3)

        with h1:

            st.metric(
                "Total Predictions",
                total_predictions
            )

        with h2:

            st.metric(
                "High Risk",
                high_predictions
            )

        with h3:

            st.metric(
                "Critical Risk",
                critical_predictions
            )

        st.markdown("---")

        st.dataframe(
            history,
            use_container_width=True
        )

        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        csv_data = history.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# PAGE 5
# RETENTION RECOMMENDATIONS
# =========================================================

elif page == "💡 Retention Recommendations":

    st.header("💡 Customer Retention Recommendations")

    st.write(
        "Generate business actions based on customer characteristics."
    )

    st.markdown("---")

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

    st.markdown("---")

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
            "🎯 Recommended Actions"
        )

        for recommendation in recommendations:

            st.success(
                f"💡 {recommendation}"
            )


# =========================================================
# PAGE 6
# REVENUE RISK
# =========================================================

elif page == "💰 Revenue Risk":

    st.header("💰 Revenue Risk Analysis")

    st.write(
        "Estimate monthly revenue associated with churned customers "
        "and customers classified as high or critical risk."
    )

    st.markdown("---")

    # -----------------------------------------------------
    # HISTORICAL CHURN REVENUE RISK
    # -----------------------------------------------------

    revenue_result = calculate_revenue_risk(
        df
    )

    if revenue_result is not None:

        churned_customers = revenue_result[
            "churned_customers"
        ]

        monthly_revenue_loss = revenue_result[
            "monthly_revenue_loss"
        ]

        average_churned_charge = revenue_result[
            "average_churned_charge"
        ]

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "🚪 Churned Customers",
                f"{churned_customers:,}"
            )

        with r2:

            st.metric(
                "💸 Monthly Revenue Loss",
                f"${monthly_revenue_loss:,.2f}"
            )

        with r3:

            st.metric(
                "💳 Avg Churned Charge",
                f"${average_churned_charge:,.2f}"
            )

        st.markdown("---")

        st.subheader(
            "📊 Historical Revenue at Risk"
        )

        revenue_chart_data = pd.DataFrame(
            {
                "Category": [
                    "Monthly Revenue Lost"
                ],
                "Revenue": [
                    monthly_revenue_loss
                ]
            }
        )

        fig_revenue = px.bar(
            revenue_chart_data,
            x="Category",
            y="Revenue",
            title="Monthly Revenue Associated with Churned Customers",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig_revenue,
            use_container_width=True
        )

    else:

        st.warning(
            "Revenue risk analysis could not be calculated. "
            "Required columns were not found."
        )

    # -----------------------------------------------------
    # HIGH / CRITICAL RISK REVENUE
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "⚠️ High & Critical Risk Revenue"
    )

    if "Risk_Level" in df.columns:

        high_risk_result = calculate_high_risk_revenue(
            df
        )

        if high_risk_result is not None:

            high_risk_customers = high_risk_result[
                "high_risk_customers"
            ]

            revenue_at_risk = high_risk_result[
                "revenue_at_risk"
            ]

            rr1, rr2 = st.columns(2)

            with rr1:

                st.metric(
                    "⚠️ High/Critical Customers",
                    f"{high_risk_customers:,}"
                )

            with rr2:

                st.metric(
                    "💰 Revenue at Risk",
                    f"${revenue_at_risk:,.2f}"
                )

        else:

            st.info(
                "High-risk revenue could not be calculated."
            )

    else:

        st.info(
            "The current dataset does not contain a Risk_Level column. "
            "High/Critical revenue becomes available after prediction "
            "results are generated and classified."
        )


# =========================================================
# PAGE 7
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.header("📈 ANN Model Performance")

    st.write(
        "Model evaluation metrics used to assess classification performance."
    )

    st.markdown("---")

    st.info(
        """
        **Important:** Model performance must be calculated from the
        ANN model's actual test-set predictions.

        The example values inside `model_performance.py` are demonstration
        data and should not be presented as the real model performance.
        """
    )

    # -----------------------------------------------------
    # OPTIONAL TEST DATA
    # -----------------------------------------------------

    st.subheader("🧪 Evaluation Status")

    st.warning(
        "Actual ANN test-set predictions are not automatically generated "
        "by the current app. The model-performance functions are ready, "
        "but real metrics require the trained ANN model and its original "
        "test dataset/features."
    )

    st.markdown("---")

    st.subheader("📋 Available Evaluation Metrics")

    metric_info = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Purpose": [
                "Overall prediction correctness",
                "Correctness of predicted churn customers",
                "Ability to identify actual churn customers",
                "Balance between precision and recall",
                "Overall classification ranking performance"
            ]
        }
    )

    st.dataframe(
        metric_info,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("🧩 Model Performance Module")

    st.code(
        """
calculate_metrics(
    y_true,
    y_pred,
    y_probability
)

create_confusion_matrix(
    y_true,
    y_pred
)

create_metrics_dataframe(
    metrics
)
        """,
        language="python"
    )

    st.info(
        "Once the actual ANN test predictions are connected, this page "
        "can display the real Accuracy, Precision, Recall, F1 Score, "
        "ROC-AUC and Confusion Matrix."
    )


# =========================================================
# PAGE 8
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About This Project")

    st.markdown(
        """
        ## AI-Powered Telecom Customer Churn Prediction & Retention Platform

        This platform uses Artificial Neural Networks (ANN), customer
        analytics and business intelligence techniques to identify telecom
        customers who may be at risk of leaving.

        ### 🎯 Main Capabilities

        • Customer churn prediction

        • Churn probability estimation

        • Customer risk classification

        • Customer analytics

        • Prediction history

        • Retention recommendations

        • Revenue risk analysis

        • Model performance monitoring

        • Business-oriented insights

        ### 🤖 Artificial Intelligence

        The system uses a trained Artificial Neural Network to estimate
        the probability that a customer may churn.

        ### 📊 Analytics

        Customer behavior can be analyzed using:

        • Contract type

        • Internet service

        • Payment method

        • Tenure

        • Monthly charges

        ### 💡 Retention Intelligence

        The platform converts customer characteristics into practical
        retention recommendations so businesses can take preventive action.

        ### 💰 Revenue Intelligence

        Revenue risk analysis estimates the monthly charges associated
        with churned customers and, when risk classifications are available,
        high and critical risk customers.

        ### 🛠 Technology Stack

        🐍 Python

        🐼 Pandas

        📊 Plotly

        🤖 TensorFlow / Keras

        🧠 Artificial Neural Network

        📈 Scikit-Learn

        🚀 Streamlit

        ### 💼 Business Value

        The goal is to help telecom companies move from reactive customer
        loss management toward proactive, AI-assisted retention.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <b>AI Telco Customer Churn Prediction</b><br>

    Customer Analytics • AI Prediction • Retention Intelligence • Revenue Risk

    </div>
    """,
    unsafe_allow_html=True
)
