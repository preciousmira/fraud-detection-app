import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit Page Setup (must come first)
# -------------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# -------------------------------
# Load model and scaler
# -------------------------------
model = joblib.load('model.pkl')
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Welcome Message
# -------------------------------
st.markdown("""
### 👋 Welcome to the Smart Fraud Detection System  
This intelligent system is designed to **analyze financial transactions** and identify potential fraud in real time.  

🔍 Using advanced **Machine Learning algorithms** like **LightGBM**, the system evaluates key transaction details — such as amount, balance, and transaction type — to determine whether a transaction is **fraudulent (1)** or **legitimate (0)**.

💡 To get started:
1. Enter transaction details in the sidebar.
2. Click **“Predict Fraud”** to see the result.
3. View the probability chart showing how confident the model is.

---
""")

st.title("💳 Fraud Detection System (LightGBM Model)")
st.markdown("""
This web app predicts whether a financial transaction is **fraudulent (1)** or **legitimate (0)**  
based on transaction details provided below.
""")

# -------------------------------
# Sidebar for user input
# -------------------------------
st.sidebar.header("Enter Transaction Details")

def user_input():
    type_ = st.sidebar.selectbox("Transaction Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.sidebar.number_input("Old Balance (Origin)", min_value=0.0, value=5000.0)
    newbalanceOrig = st.sidebar.number_input("New Balance (Origin)", min_value=0.0, value=4000.0)
    oldbalanceDest = st.sidebar.number_input("Old Balance (Destination)", min_value=0.0, value=1000.0)
    newbalanceDest = st.sidebar.number_input("New Balance (Destination)", min_value=0.0, value=2000.0)
    isFlaggedFraud = st.sidebar.selectbox("Is Flagged Fraud (1=True, 0=False)", [0, 1])
    hour_sin = st.sidebar.slider("Hour (sin)", -1.0, 1.0, 0.0)
    hour_cos = st.sidebar.slider("Hour (cos)", -1.0, 1.0, 0.0)
    day_sin = st.sidebar.slider("Day (sin)", -1.0, 1.0, 0.0)
    day_cos = st.sidebar.slider("Day (cos)", -1.0, 1.0, 0.0)

    # Encode transaction type
    type_mapping = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}
    type_encoded = type_mapping[type_]

    data = {
        'type': type_encoded,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': isFlaggedFraud,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos
    }

    return pd.DataFrame([data])

input_df = user_input()

st.subheader("🧾 Transaction Data Preview")
st.dataframe(input_df)

# -------------------------------
# Scaling numeric features
# -------------------------------
scaled_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaled_df = input_df.copy()
scaled_df[scaled_features] = scaler.transform(input_df[scaled_features])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(scaled_df)[0]
    probability = model.predict_proba(scaled_df)[0][1]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 ALERT: This transaction is likely **FRAUDULENT!** (Probability: {probability:.2%})")
    else:
        st.success(f"✅ This transaction appears **LEGITIMATE.** (Probability: {probability:.2%})")

# -------------------------------
# Fraud Trend Visualization
# -------------------------------
st.subheader("📊 Fraud Trend Visualization")

np.random.seed(42)
dates = pd.date_range(start=datetime.today() - timedelta(days=30), periods=30)
fraud_counts = np.random.randint(0, 50, size=30)
legit_counts = np.random.randint(50, 150, size=30)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(dates, fraud_counts, label="Fraudulent Transactions", linewidth=2)
ax.plot(dates, legit_counts, label="Legitimate Transactions", linewidth=2)
ax.set_title("Fraud Trends Over Time", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Number of Transactions")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -------------------------------
# Footer (will now display correctly)
# -------------------------------
st.markdown("---")
st.markdown("""
### 📘 **Project Information**  
**Title:** Design and Implementation of Machine Learning in Fraud Detection and Risk Management System  
**Student:** Onyemachi Precious  
**Supervisor:** Ugorji Clinton Chikezie  
**Institution:** Abia State Polythenic  
""")

st.caption("Developed by Onyemachi Precious | Final Year Project - Fraud Detection System")

