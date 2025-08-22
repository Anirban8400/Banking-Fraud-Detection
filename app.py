import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Fraud Detection App")

st.write("Enter transaction details below:")

# Numeric inputs (raw values, unscaled)
amount = st.number_input("Amount", value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", value=5000.0)
isFlaggedFraud = st.selectbox("Is Flagged Fraud", [0, 1])
is_merchant_dest = st.selectbox("Is receiver a merchant (destination account starts with M)?", [0, 1])
oldBalanceOrig = st.number_input("Balance of origin before Transaction", value=0.0)
newBalanceOrig = st.number_input("New Balance of origin", value=0.0)
oldBalanceDest = st.number_input("Balance of receiver before transaction", value=0.0)
newBalanceDest = st.number_input("New Balance of receiver", value=0.0)
day = st.slider("Day", min_value=0, max_value=30, value=0)

# One-hot encoded type
type_options = ["CASH_IN", "CASH_OUT", "DEBIT", "TRANSFER"]
txn_type = st.selectbox("Transaction Type", type_options)

type_dict = {t: 0 for t in type_options}
type_dict[txn_type] = 1

errorOrig=oldBalanceOrig-amount-newBalanceOrig
errorDest=oldBalanceDest-amount-newBalanceDest

# Create dataframe (raw values)
input_df = pd.DataFrame([{
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "isFlaggedFraud": isFlaggedFraud,
    "is_merchant_dest": is_merchant_dest,
    "errorOrig": errorOrig,
    "errorDest": errorDest,
    "log(amount)": np.log1p(amount),
    "day": day,
    "type_CASH_IN": type_dict["CASH_IN"],
    "type_CASH_OUT": type_dict["CASH_OUT"],
    "type_DEBIT": type_dict["DEBIT"],
    "type_TRANSFER": type_dict["TRANSFER"]
}])

# Apply same StandardScaler
scaled_input = scaler.transform(input_df)

if st.button("Predict Fraud"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Predicted as FRAUD (probability: {proba:.2f})")
    else:
        st.success(f"✅ Predicted as NON-FRAUD (probability: {proba:.2f})")
