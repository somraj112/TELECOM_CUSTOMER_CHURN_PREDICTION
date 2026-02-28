import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD SAVED FILES
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("AI Customer Churn Prediction System")

st.write("Enter customer details to predict churn probability.")

# USER INPUT FORM

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72)
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", 
                       ["Electronic check", 
                        "Mailed check", 
                        "Bank transfer (automatic)", 
                        "Credit card (automatic)"])

monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

# PREDICT BUTTON

if st.button("Predict Churn"):

    # Create dictionary of inputs
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # One-hot encode (same as training)
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{probability*100:.2f}%")

    st.progress(probability)

    if probability > 0.7:
        st.error("High Risk Customer")
        st.write("Suggested Action:")
        st.write("- Offer long-term discounted contract")
        st.write("- Provide loyalty benefits")
        st.write("- Assign customer success agent")

    elif probability > 0.4:
        st.warning("Medium Risk Customer")
        st.write("Suggested Action:")
        st.write("- Provide personalized offers")
        st.write("- Monitor usage pattern")

    else:
        st.success("Low Risk Customer")
        st.write("Customer likely to stay.")