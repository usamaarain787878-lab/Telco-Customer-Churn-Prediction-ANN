# customer_analytics.py

import pandas as pd
import plotly.express as px


def churn_by_contract(df):
    """
    Analyze customer churn according to contract type.
    """

    if "Contract" not in df.columns or "Churn" not in df.columns:
        return None

    data = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    fig = px.bar(
        data,
        barmode="group",
        title="Customer Churn by Contract Type",
        labels={
            "value": "Number of Customers",
            "Contract": "Contract Type"
        }
    )

    return fig


def churn_by_internet(df):
    """
    Analyze customer churn according to internet service.
    """

    if "InternetService" not in df.columns or "Churn" not in df.columns:
        return None

    data = pd.crosstab(
        df["InternetService"],
        df["Churn"]
    )

    fig = px.bar(
        data,
        barmode="group",
        title="Customer Churn by Internet Service",
        labels={
            "value": "Number of Customers",
            "InternetService": "Internet Service"
        }
    )

    return fig


def churn_by_payment(df):
    """
    Analyze customer churn according to payment method.
    """

    if "PaymentMethod" not in df.columns or "Churn" not in df.columns:
        return None

    data = pd.crosstab(
        df["PaymentMethod"],
        df["Churn"]
    )

    fig = px.bar(
        data,
        barmode="group",
        title="Customer Churn by Payment Method",
        labels={
            "value": "Number of Customers",
            "PaymentMethod": "Payment Method"
        }
    )

    return fig


def churn_by_tenure(df):
    """
    Analyze customer churn according to tenure.
    """

    if "tenure" not in df.columns or "Churn" not in df.columns:
        return None

    data = df.copy()

    data["tenure"] = pd.to_numeric(
        data["tenure"],
        errors="coerce"
    )

    fig = px.histogram(
        data,
        x="tenure",
        color="Churn",
        nbins=20,
        title="Customer Churn by Tenure",
        labels={
            "tenure": "Tenure (Months)",
            "count": "Customers"
        }
    )

    return fig


def churn_by_monthly_charges(df):
    """
    Analyze churn according to monthly charges.
    """

    if "MonthlyCharges" not in df.columns or "Churn" not in df.columns:
        return None

    data = df.copy()

    data["MonthlyCharges"] = pd.to_numeric(
        data["MonthlyCharges"],
        errors="coerce"
    )

    fig = px.histogram(
        data,
        x="MonthlyCharges",
        color="Churn",
        nbins=20,
        title="Customer Churn by Monthly Charges",
        labels={
            "MonthlyCharges": "Monthly Charges"
        }
    )

    return fig


# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":

    df = pd.read_csv("Cleaned_Dataset.csv")

    print("Customer Analytics Module")

    print("\nDataset Shape:")
    print(df.shape)

    if "Contract" in df.columns:
        print("\nContract Types:")
        print(df["Contract"].value_counts())

    if "InternetService" in df.columns:
        print("\nInternet Services:")
        print(df["InternetService"].value_counts())

    if "PaymentMethod" in df.columns:
        print("\nPayment Methods:")
        print(df["PaymentMethod"].value_counts())
