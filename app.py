# ============================================================
# Customer Churn Prediction System - Streamlit App
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide"
)


# ------------------------------------------------------------
# Load Model and Metadata
# ------------------------------------------------------------

@st.cache_resource
def load_model_and_metadata():
    model = joblib.load("models/customer_churn_prediction_pipeline.pkl")

    with open("models/model_metadata.json", "r") as file:
        metadata = json.load(file)

    return model, metadata


model, metadata = load_model_and_metadata()
selected_threshold = metadata["selected_threshold"]


# ------------------------------------------------------------
# Feature Engineering Function
# ------------------------------------------------------------

def create_customer_churn_features(dataframe):
    data = dataframe.copy()

    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
    ).astype("object")

    data["HasInternetService"] = np.where(
        data["InternetService"] == "No", "No", "Yes"
    )

    data["IsMonthToMonth"] = np.where(
        data["Contract"] == "Month-to-month", "Yes", "No"
    )

    support_service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport"
    ]

    data["NumberOfSupportServices"] = data[support_service_cols].apply(
        lambda row: int(sum(row == "Yes")),
        axis=1
    )

    data["AverageChargesPerTenure"] = (
        data["TotalCharges"] / data["tenure"].replace(0, 1)
    )

    return data


# ------------------------------------------------------------
# Business Logic
# ------------------------------------------------------------

def categorize_churn_risk(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def recommend_business_action(risk_category):
    if risk_category == "High Risk":
        return "Immediate retention offer or customer support follow-up"
    elif risk_category == "Medium Risk":
        return "Monitor customer and offer personalized engagement"
    else:
        return "Maintain regular customer relationship"


def predict_customer_churn(customer_data):
    prepared_data = create_customer_churn_features(customer_data)

    churn_probability = model.predict_proba(prepared_data)[:, 1][0]
    predicted_churn = int(churn_probability >= selected_threshold)

    risk_category = categorize_churn_risk(churn_probability)
    recommended_action = recommend_business_action(risk_category)

    return churn_probability, predicted_churn, risk_category, recommended_action


# ------------------------------------------------------------
# App Header
# ------------------------------------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    """
    This app predicts whether a customer is likely to churn based on customer profile,
    service usage, billing information, and contract details.
    """
)

st.info(
    f"Final Model: {metadata['best_model']} | "
    f"Selected Threshold: {selected_threshold:.2f} | "
    f"ROC-AUC: {metadata['final_metrics']['roc_auc']:.3f} | "
    f"Recall: {metadata['final_metrics']['recall']:.3f}"
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
    step=1.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0,
    step=10.0
)


# ------------------------------------------------------------
# Create Input DataFrame
# ------------------------------------------------------------

input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})


# ------------------------------------------------------------
# Main App Layout
# ------------------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Input Preview")
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("Model Information")

    st.write(f"**Best Model:** {metadata['best_model']}")
    st.write(f"**Business Threshold:** {selected_threshold:.2f}")
    st.write(f"**ROC-AUC:** {metadata['final_metrics']['roc_auc']:.3f}")
    st.write(f"**Recall:** {metadata['final_metrics']['recall']:.3f}")
    st.write(f"**F1 Score:** {metadata['final_metrics']['f1_score']:.3f}")


# ------------------------------------------------------------
# Prediction Button
# ------------------------------------------------------------

if st.button("Predict Churn Risk"):
    probability, prediction, risk_category, action = predict_customer_churn(input_data)

    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Churn Probability", f"{probability * 100:.2f}%")

    with result_col2:
        st.metric("Predicted Churn", "Yes" if prediction == 1 else "No")

    with result_col3:
        st.metric("Risk Category", risk_category)

    if risk_category == "High Risk":
        st.error(f"🚨 {risk_category}: {action}")
    elif risk_category == "Medium Risk":
        st.warning(f"⚠️ {risk_category}: {action}")
    else:
        st.success(f"✅ {risk_category}: {action}")

    st.write("### Business Interpretation")

    if risk_category == "High Risk":
        st.write(
            "This customer has a high probability of churn. "
            "The business should prioritize retention action, such as a personalized offer, "
            "service follow-up, or customer support intervention."
        )
    elif risk_category == "Medium Risk":
        st.write(
            "This customer shows moderate churn risk. "
            "The business should monitor customer behavior and consider targeted engagement."
        )
    else:
        st.write(
            "This customer is currently low risk. "
            "The business can maintain regular relationship management."
        )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")
st.caption(
    "End-to-End Machine Learning Project | Data Cleaning | Feature Engineering | "
    "Model Comparison | Threshold Tuning | Business Risk Prediction | Streamlit Deployment"
)