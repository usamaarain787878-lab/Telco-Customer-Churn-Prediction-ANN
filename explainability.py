# explainability.py

def explain_customer_risk(
    contract,
    monthly_charges,
    tenure,
    tech_support,
    churn_probability
):
    """
    Generate business-friendly explanations
    for customer churn risk.
    """

    factors = []
    actions = []

    # Contract factor
    if contract == "Month-to-month":
        factors.append(
            "Month-to-month contract increases churn risk."
        )
        actions.append(
            "Offer a long-term contract with a loyalty discount."
        )

    # Monthly charges factor
    if monthly_charges > 70:
        factors.append(
            "High monthly charges may increase customer dissatisfaction."
        )
        actions.append(
            "Consider a personalized pricing discount or lower-cost plan."
        )

    # Tenure factor
    if tenure < 12:
        factors.append(
            "Low customer tenure indicates a relatively new customer."
        )
        actions.append(
            "Provide a new-customer retention offer."
        )

    # Technical support factor
    if tech_support == "No":
        factors.append(
            "Customer does not have technical support."
        )
        actions.append(
            "Offer technical support or a support package."
        )

    # Default factor
    if not factors:
        factors.append(
            "No major predefined churn risk factors were detected."
        )

    # Risk explanation
    if churn_probability >= 80:
        risk_explanation = (
            "Critical risk: immediate retention action is recommended."
        )

    elif churn_probability >= 60:
        risk_explanation = (
            "High risk: proactive customer retention action is recommended."
        )

    elif churn_probability >= 30:
        risk_explanation = (
            "Medium risk: monitor the customer and consider preventive action."
        )

    else:
        risk_explanation = (
            "Low risk: customer currently shows relatively low churn risk."
        )

    # Default action
    if not actions:
        actions.append(
            "Continue monitoring customer satisfaction and engagement."
        )

    return {
        "risk_explanation": risk_explanation,
        "risk_factors": factors,
        "recommended_actions": actions
    }


# ---------------------------------------------------------
# Demo
# ---------------------------------------------------------

if __name__ == "__main__":

    result = explain_customer_risk(
        contract="Month-to-month",
        monthly_charges=85,
        tenure=5,
        tech_support="No",
        churn_probability=84
    )

    print("========================================")
    print("AI CUSTOMER RISK EXPLANATION")
    print("========================================")

    print("\nRisk Explanation:")
    print(result["risk_explanation"])

    print("\nRisk Factors:")

    for factor in result["risk_factors"]:
        print("-", factor)

    print("\nRecommended Actions:")

    for action in result["recommended_actions"]:
        print("-", action)
