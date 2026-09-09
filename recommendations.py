# recommendations.py

def get_recommendations(contract, monthly_charges, tenure, tech_support):
    recommendations = []

    # Contract recommendation
    if contract == "Month-to-month":
        recommendations.append(
            "Offer the customer a long-term contract with a special discount."
        )

    # Monthly charges recommendation
    if monthly_charges > 70:
        recommendations.append(
            "Consider offering a personalized discount to reduce monthly cost."
        )

    # Tenure recommendation
    if tenure < 12:
        recommendations.append(
            "Provide a new-customer retention offer to encourage long-term loyalty."
        )

    # Technical support recommendation
    if tech_support == "No":
        recommendations.append(
            "Offer technical support or a support package to improve customer experience."
        )

    # Default recommendation
    if len(recommendations) == 0:
        recommendations.append(
            "Continue providing good service and monitor customer satisfaction."
        )

    return recommendations


# Example
if __name__ == "__main__":

    result = get_recommendations(
        contract="Month-to-month",
        monthly_charges=85,
        tenure=5,
        tech_support="No"
    )

    print("AI Retention Recommendations:")

    for recommendation in result:
        print("-", recommendation)
