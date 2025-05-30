import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("best_log_reg_model.pkl")

st.title("Telecom Customer Churn Prediction")
st.write("Enter customer information to predict whether they will churn.")

# Input fields (adjust based on model feature requirements)
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# One-hot encoding of categorical variables
contract_map = {
    "Month-to-month": [1, 0, 0],
    "One year": [0, 1, 0],
    "Two year": [0, 0, 1]
}

internet_map = {
    "DSL": [1, 0, 0],
    "Fiber optic": [0, 1, 0],
    "No": [0, 0, 1]
}

payment_map = {
    "Electronic check": [1, 0, 0, 0],
    "Mailed check": [0, 1, 0, 0],
    "Bank transfer (automatic)": [0, 0, 1, 0],
    "Credit card (automatic)": [0, 0, 0, 1]
}

features = [tenure, monthly_charges] + \
           contract_map[contract_type] + \
           internet_map[internet_service] + \
           payment_map[payment_method]

input_df = pd.DataFrame([features])

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"The customer is likely to churn (Probability: {proba:.2f})")
    else:
        st.success(f"The customer is likely to stay (Probability: {1 - proba:.2f})")
