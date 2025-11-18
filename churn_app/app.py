import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ---------- Load model & metadata ----------
MODEL_PATH = Path("../models/xgb_churn_pipeline.pkl")
META_PATH = Path("../models/metadata.json")

xgb_model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())
THRESHOLD = meta["threshold"]

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

st.title("📉 Customer Churn Prediction App")
st.write(
    "This app uses an XGBoost model trained on the Telco Customer Churn dataset "
    "to estimate the probability that a customer will churn."
)

st.markdown("---")

# ---------- Input form ----------
st.subheader("📋 Enter customer details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox(
        "Online Security", ["No internet service", "No", "Yes"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["No internet service", "No", "Yes"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection", ["No internet service", "No", "Yes"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["No internet service", "No", "Yes"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["No internet service", "No", "Yes"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["No internet service", "No", "Yes"]
    )

st.markdown("---")

col4, col5 = st.columns(2)

with col4:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col5:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, max_value=100000.0, value=70.0
    )

# ---------- Build a single-row DataFrame ----------
input_dict = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

input_df = pd.DataFrame([input_dict])

st.markdown("---")

# ---------- Prediction ----------
if st.button("Predict churn risk"):
    proba = xgb_model.predict_proba(input_df)[0, 1]
    will_churn = proba >= THRESHOLD

    st.metric("Churn probability", f"{proba:.3f}")

    if will_churn:
        st.error(
            f"⚠️ This customer is **LIKELY TO CHURN** (threshold = {THRESHOLD:.2f})."
        )
    else:
        st.success(
            f"✅ This customer is **NOT LIKELY TO CHURN** (threshold = {THRESHOLD:.2f})."
        )

    st.caption(
        "Note: The decision is based on the tuned threshold that prioritizes recall "
        "for churners."
    )
