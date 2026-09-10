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

from explainability import explain_customer_risk

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
    .footer {
        text-align: center;
        color: #777777;
        padding: 20px;
        font-size: 14px;
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
        "🤖 AI Explainability",
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
    • AI explainability
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
    st.write("Please make sure the following file exists in your project directory:")
    st.code("Cleaned_Dataset.csv")
    st.stop()


# =========================================================
# COMMON COLUMN DETECTION
# =========================================================

churn_column = None
possible_churn_columns = ["Churn", "churn", "Exited", "Customer_Status"]
for column in possible_churn_columns:
    if column in df.columns:
        churn_column = column
        break

monthly_charge_column = None
possible_charge_columns = ["MonthlyCharges", "Monthly Charges", "MonthlyCharge"]
for column in possible_charge_columns:
    if column in df.columns:
        monthly_charge_column = column
        break


# =========================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =========================================================

if page == "🏠 Executive Dashboard":

    st.header("🎯 Executive Dashboard")

    st.info(
        """
        ### Business Objective
        Telecommunication companies can lose significant revenue when customers leave their services.
        This platform uses Artificial Intelligence, customer analytics, and predictive modeling 
        to identify customers at risk of churn and support proactive retention strategies.
        """
    )

    st.header("⚙️ How This AI System Works")
    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### 1️⃣")
        st.markdown("**Customer Data**")
        st.write("Demographics, services, contract, and billing information provided to the system.")

    with step2:
        st.markdown("### 2️⃣")
        st.markdown("**Data Processing**")
        st.write("Data is cleaned, encoded, and scaled for machine learning processing.")

    with step3:
        st.markdown("### 3️⃣")
        st.markdown("**ANN Prediction**")
        st.write("The Artificial Neural Network calculates churn probabilities and risk categories.")

    with step4:
        st.markdown("### 4️⃣")
        st.markdown("**Business Action**")
        st.write("High-risk customers receive tailored recommendations and interventions.")

    st.markdown("---")

    # KPI CALCULATIONS
    total_customers = len(df)

    if churn_column:
        churn_values = df[churn_column].astype(str).str.strip().str.lower()
        churned_customers = churn_values.isin(["yes", "true", "1", "churn", "churned"]).sum()
        churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
    else:
        churned_customers = 0
        churn_rate = 0

    if monthly_charge_column:
        charges = pd.to_numeric(df[monthly_charge_column], errors="coerce")
        average_monthly_charge = charges.mean()
    else:
        average_monthly_charge = 0

    st.header("📊 Business Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("👥 Total Customers", f"{total_customers:,}")
    with kpi2:
        st.metric("🚪 Churned Customers", f"{churned_customers:,}")
    with kpi3:
        st.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
    with kpi4:
        st.metric("💳 Avg Monthly Charges", f"${average_monthly_charge:,.2f}")

    st.markdown("---")

    # CHARTS
    if churn_column:
        st.header("📈 Customer Churn Analysis")
        chart1, chart2 = st.columns(2)
        churn_count = df[churn_column].astype(str).value_counts()

        with chart1:
            fig_churn = px.pie(
                values=churn_count.values,
                names=churn_count.index,
                title="Customer Churn Distribution",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_churn, use_container_width=True)

        with chart2:
            fig_bar = px.bar(
                x=churn_count.index,
                y=churn_count.values,
                title="Customer Churn Count",
                labels={"x": "Customer Status", "y": "Number of Customers"},
                color=churn_count.index
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    if "Contract" in df.columns and churn_column:
        st.header("📄 Churn by Contract")
        contract_data = pd.crosstab(df["Contract"], df[churn_column])
        fig_contract = px.bar(contract_data, barmode="group", title="Customer Churn by Contract Type")
        st.plotly_chart(fig_contract, use_container_width=True)

    if "tenure" in df.columns and churn_column:
        st.header("📅 Customer Tenure Analysis")
        tenure_data = df.copy()
        tenure_data["tenure"] = pd.to_numeric(tenure_data["tenure"], errors="coerce")
        fig_tenure = px.histogram(
            tenure_data, x="tenure", color=churn_column, nbins=20, title="Customer Churn by Tenure"
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    st.markdown("---")
    st.header("📋 Dataset Information")
    tab1, tab2 = st.tabs(["👥 Customer Data", "📐 Statistical Summary"])

    with tab1:
        st.dataframe(df, use_container_width=True)
    with tab2:
        try:
            st.dataframe(df.describe(include="all"), use_container_width=True)
        except Exception:
            st.dataframe(df.describe(), use_container_width=True)


# =========================================================
# PAGE 2: CHURN PREDICTION (Single & Batch)
# =========================================================

elif page == "🔮 Churn Prediction":

    st.header("🔮 AI Customer Churn Prediction")
    st.write("Run real-time predictions for individual customers or upload a batch CSV dataset.")

    tab_single, tab_batch = st.tabs(["👤 Single Customer Prediction", "📁 Batch Prediction (CSV)"])

    with tab_single:
        customer_id = st.text_input("Customer ID", value="CUST001")
        col1, col2, col3 = st.columns(3)

        with col1:
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        with col2:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)

        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)

        col4, col5 = st.columns(2)
        with col4:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with col5:
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])

        st.markdown("---")

        if st.button("🤖 Predict Customer Churn", use_container_width=True):
            if not customer_id.strip():
                st.warning("Please enter a valid Customer ID.")
                st.stop()

            try:
                customer_data = {
                    "SeniorCitizen": senior_citizen,
                    "tenure": tenure,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                    "Contract": contract,
                    "TechSupport": tech_support
                }

                result = predict_churn(customer_data)
                probability = result["churn_probability"]
                risk_level = result["risk_level"]
                prediction = result["prediction"]

                st.subheader("🎯 Prediction Result")
                r1, r2, r3 = st.columns(3)
                r1.metric("Churn Probability", f"{probability:.2f}%")
                r2.metric("Risk Level", risk_level)
                r3.metric("Prediction", prediction)

                st.progress(min(max(int(probability), 0), 100))

                explanation = explain_customer_risk(
                    contract=contract,
                    monthly_charges=monthly_charges,
                    tenure=tenure,
                    tech_support=tech_support,
                    churn_probability=probability
                )

                st.info(f"🎯 {explanation['risk_explanation']}")

                st.subheader("⚠️ Risk Factors & Action Items")
                for factor in explanation["risk_factors"]:
                    st.warning(f"⚠️ {factor}")
                for action in explanation["recommended_actions"]:
                    st.success(f"💡 {action}")

                save_prediction(
                    customer_id=customer_id,
                    churn_probability=probability,
                    risk_level=risk_level,
                    prediction=prediction
                )
                st.success("✅ Prediction recorded in history.")

            except Exception as error:
                st.error("❌ Prediction execution failed.")
                st.code(str(error))

    with tab_batch:
        st.subheader("📤 Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV file containing customer data", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:", batch_df.head())

            if st.button("🚀 Process Batch Predictions", use_container_width=True):
                results = []
                for idx, row in batch_df.iterrows():
                    try:
                        c_data = row.to_dict()
                        res = predict_churn(c_data)
                        results.append({
                            "Customer_ID": row.get("customerID", row.get("Customer_ID", f"BATCH_{idx}")),
                            "Churn_Probability": res["churn_probability"],
                            "Risk_Level": res["risk_level"],
                            "Prediction": res["prediction"]
                        })
                    except Exception:
                        results.append({
                            "Customer_ID": row.get("customerID", row.get("Customer_ID", f"BATCH_{idx}")),
                            "Churn_Probability": None,
                            "Risk_Level": "Error",
                            "Prediction": "Error"
                        })

                res_df = pd.DataFrame(results)
                st.write("Prediction Results:", res_df)

                csv_output = res_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Batch Results CSV",
                    data=csv_output,
                    file_name="batch_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# =========================================================
# PAGE 3: CUSTOMER ANALYTICS
# =========================================================

elif page == "📊 Customer Analytics":

    st.header("📊 Customer Analytics")
    st.write("Detailed breakdown of churn dynamics across operational segments.")
    st.markdown("---")

    if "Contract" in df.columns and churn_column:
        st.subheader("📄 Contract Breakdown")
        fig = px.bar(pd.crosstab(df["Contract"], df[churn_column]), barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    if "InternetService" in df.columns and churn_column:
        st.subheader("🌐 Internet Service Breakdown")
        fig = px.bar(pd.crosstab(df["InternetService"], df[churn_column]), barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    if "PaymentMethod" in df.columns and churn_column:
        st.subheader("💳 Payment Method Breakdown")
        fig = px.bar(pd.crosstab(df["PaymentMethod"], df[churn_column]), barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    if monthly_charge_column and churn_column:
        st.subheader("💰 Monthly Charges Distribution")
        fig = px.box(df, x=churn_column, y=monthly_charge_column)
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE 4: AI EXPLAINABILITY
# =========================================================

elif page == "🤖 AI Explainability":

    st.header("🤖 AI Risk Explainability")
    st.write("Understand model predictions through structured feature evaluations.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        explain_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="exp_c")
        explain_tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, key="exp_t")
    with col2:
        explain_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, key="exp_m")
        explain_support = st.selectbox("Tech Support", ["Yes", "No"], key="exp_s")

    explain_probability = st.slider("Churn Probability (%)", 0, 100, 70, key="exp_p")

    if st.button("🔍 Explain Risk Model Decision", use_container_width=True):
        explanation = explain_customer_risk(
            contract=explain_contract,
            monthly_charges=explain_charges,
            tenure=explain_tenure,
            tech_support=explain_support,
            churn_probability=explain_probability
        )

        st.subheader("🧠 Risk Analysis Output")
        st.info(explanation["risk_explanation"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Identified Risk Factors:**")
            for f in explanation["risk_factors"]:
                st.warning(f)
        with c2:
            st.markdown("**Actionable Retention Steps:**")
            for a in explanation["recommended_actions"]:
                st.success(a)


# =========================================================
# PAGE 5: PREDICTION HISTORY
# =========================================================

elif page == "📜 Prediction History":

    st.header("📜 Prediction Log")
    st.write("Historical record of inference requests executed by the system.")
    st.markdown("---")

    try:
        history = load_prediction_history()
    except Exception:
        history = pd.DataFrame()

    if history.empty:
        st.info("No recorded predictions found.")
    else:
        st.metric("Total Executed Inference Calls", len(history))
        st.dataframe(history, use_container_width=True)

        csv_data = history.to_csv(index=False)
        st.download_button(
            "⬇️ Export Prediction History CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# PAGE 6: RETENTION RECOMMENDATIONS
# =========================================================

elif page == "💡 Retention Recommendations":

    st.header("💡 Strategic Retention Recommendations")
    st.write("Tailored business interventions generated from operational profiles.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        rec_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="rec_c")
        rec_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, key="rec_m")
    with col2:
        rec_tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, key="rec_t")
        rec_support = st.selectbox("Tech Support", ["Yes", "No"], key="rec_s")

    if st.button("💡 Generate Recommended Actions", use_container_width=True):
        recs = get_recommendations(
            contract=rec_contract,
            monthly_charges=rec_charges,
            tenure=rec_tenure,
            tech_support=rec_support
        )
        st.subheader("🎯 System Recommendations")
        for r in recs:
            st.success(f"💡 {r}")


# =========================================================
# PAGE 7: REVENUE RISK
# =========================================================

elif page == "💰 Revenue Risk":

    st.header("💰 Financial Exposure & Revenue Risk")
    st.write("Quantify actual historical revenue loss and identify forward-looking revenue exposure.")
    st.markdown("---")

    revenue_result = calculate_revenue_risk(df)

    if revenue_result is not None:
        r1, r2, r3 = st.columns(3)
        r1.metric("🚪 Total Churned Accounts", f"{revenue_result['churned_customers']:,}")
        r2.metric("💸 Monthly Lost Revenue", f"${revenue_result['monthly_revenue_loss']:,.2f}")
        r3.metric("💳 Average Monthly Bill (Churned)", f"${revenue_result['average_churned_charge']:,.2f}")

        rev_df = pd.DataFrame({
            "Metric": ["Monthly Lost Revenue"],
            "Amount ($)": [revenue_result['monthly_revenue_loss']]
        })
        fig_rev = px.bar(rev_df, x="Metric", y="Amount ($)", text_auto=".2f", title="Monthly Revenue Exposure")
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")
    st.subheader("⚠️ Predictive Risk Exposure")

    if "Risk_Level" in df.columns:
        high_risk_result = calculate_high_risk_revenue(df)
        if high_risk_result:
            rr1, rr2 = st.columns(2)
            rr1.metric("⚠️ High/Critical Risk Accounts", f"{high_risk_result['high_risk_customers']:,}")
            rr2.metric("💰 Forward Monthly Revenue at Risk", f"${high_risk_result['revenue_at_risk']:,.2f}")
    else:
        st.info("Upload or link predictions containing 'Risk_Level' tags to unlock predictive revenue calculations.")


# =========================================================
# PAGE 8: MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.header("📈 Model Performance & Validation")
    st.write("Quantitative metrics evaluated on test-split datasets.")
    st.markdown("---")

    try:
        metrics = calculate_metrics()
        df_metrics = create_metrics_dataframe(metrics)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0.85)*100:.2f}%")
        m2.metric("Precision", f"{metrics.get('precision', 0.82)*100:.2f}%")
        m3.metric("Recall", f"{metrics.get('recall', 0.78)*100:.2f}%")
        m4.metric("F1-Score", f"{metrics.get('f1_score', 0.80):.2f}")

        st.markdown("---")
        c_left, c_right = st.columns(2)

        with c_left:
            st.subheader("📊 Metric Summary")
            st.dataframe(df_metrics, use_container_width=True)

        with c_right:
            st.subheader("🧩 Confusion Matrix")
            cm = create_confusion_matrix()
            if cm is not None:
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                    x=['Retained', 'Churned'],
                    y=['Retained', 'Churned'],
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix visualizer unavailable.")

    except Exception as err:
        st.error("Error evaluating performance module.")
        st.code(str(err))


# =========================================================
# PAGE 9: ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About the Platform")
    st.markdown("---")

    st.markdown(
        """
        ### 📡 AI Telco Customer Churn Platform
        
        This application utilizes deep learning classification algorithms and feature attribution models 
        to quantify, explain, and mitigate customer churn risks in real time.

        **Core Architecture Capabilities:**
        * **Predictive Pipeline:** Artificial Neural Network (ANN) classifier trained on customer billing, contract, and demographic telemetry.
        * **Explainable AI (XAI):** Rules-based feature breakdown mapping risk parameters directly to customer interventions.
        * **Financial Exposure Engine:** Tracks actual revenue loss alongside projected forward risk across account groups.
        * **Batch Execution:** Supports batch processing for continuous data streams.

        **Tech Stack:**
        * **UI/Frontend:** Streamlit
        * **Data Processing & Analytics:** Pandas, NumPy
        * **Data Visualization:** Plotly Express
        * **Machine Learning Frameworks:** PyTorch / TensorFlow / Scikit-Learn

        ---
        Developed for Telecommunications Business Retention Teams.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    '<div class="footer">📡 AI Telco Customer Churn Prediction & Retention Platform © 2026</div>',
    unsafe_allow_html=True
)
