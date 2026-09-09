# revenue_risk.py

import pandas as pd


def calculate_revenue_risk(
    df,
    churn_column="Churn",
    monthly_charge_column="MonthlyCharges"
):
    """
    Calculate potential monthly revenue at risk
    from customers who have already churned.
    """

    # Check required columns
    if churn_column not in df.columns:
        return None

    if monthly_charge_column not in df.columns:
        return None

    data = df.copy()

    # Convert monthly charges to numbers
    data[monthly_charge_column] = pd.to_numeric(
        data[monthly_charge_column],
        errors="coerce"
    )

    # Remove missing charges
    data = data.dropna(
        subset=[monthly_charge_column]
    )

    # Find churned customers
    churned = data[
        data[churn_column]
        .astype(str)
        .str.lower()
        .isin(["yes", "true", "1", "churn", "churned"])
    ]

    # Number of churned customers
    churned_customers = len(churned)

    # Revenue lost per month
    monthly_revenue_loss = churned[
        monthly_charge_column
    ].sum()

    # Average monthly charge of churned customers
    if churned_customers > 0:

        average_churned_charge = churned[
            monthly_charge_column
        ].mean()

    else:

        average_churned_charge = 0

    return {
        "churned_customers": churned_customers,
        "monthly_revenue_loss": monthly_revenue_loss,
        "average_churned_charge": average_churned_charge
    }


def calculate_high_risk_revenue(
    df,
    risk_column="Risk_Level",
    monthly_charge_column="MonthlyCharges"
):
    """
    Calculate potential revenue at risk from
    customers classified as High or Critical risk.
    """

    if risk_column not in df.columns:
        return None

    if monthly_charge_column not in df.columns:
        return None

    data = df.copy()

    data[monthly_charge_column] = pd.to_numeric(
        data[monthly_charge_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[monthly_charge_column]
    )

    # High and Critical risk customers
    high_risk = data[
        data[risk_column]
        .astype(str)
        .str.lower()
        .isin(["high", "critical"])
    ]

    high_risk_customers = len(high_risk)

    revenue_at_risk = high_risk[
        monthly_charge_column
    ].sum()

    return {
        "high_risk_customers": high_risk_customers,
        "revenue_at_risk": revenue_at_risk
    }


# =========================================================
# EXAMPLE
# =========================================================

if __name__ == "__main__":

    df = pd.read_csv(
        "Cleaned_Dataset.csv"
    )

    result = calculate_revenue_risk(df)

    if result:

        print("================================")
        print("REVENUE RISK ANALYSIS")
        print("================================")

        print(
            "Churned Customers:",
            result["churned_customers"]
        )

        print(
            "Monthly Revenue Loss: $",
            round(
                result["monthly_revenue_loss"],
                2
            )
        )

        print(
            "Average Churned Customer Charge: $",
            round(
                result["average_churned_charge"],
                2
            )
        )

    else:

        print(
            "Required columns were not found."
        )
