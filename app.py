import streamlit as st
import numpy as np
import joblib

rf      = joblib.load('random_forest_model.pkl')
scaler  = joblib.load('scaler.pkl')
le1     = joblib.load('le_transaction_type.pkl')
le2     = joblib.load('le_location.pkl')

st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="centered")
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Enter transaction details to check if it's **Fraud or Legit**")
st.divider()

col1, col2 = st.columns(2)
with col1:
    amount           = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1000.0)
    transaction_type = st.selectbox("Transaction Type", le1.classes_)
    hour             = st.slider("Hour of Transaction", 0, 23, 12)
with col2:
    merchant_id = st.number_input("Merchant ID", min_value=0, value=100)
    location    = st.selectbox("Location", le2.classes_)
    day         = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

st.divider()

if st.button("🔎 Check Transaction", use_container_width=True):
    t_encoded    = le1.transform([transaction_type])[0]
    l_encoded    = le2.transform([location])[0]
    input_data   = np.array([[amount, merchant_id, t_encoded, l_encoded, hour, day]])
    input_scaled = scaler.transform(input_data)
    prob         = rf.predict_proba(input_scaled)[0][1]
    prediction   = rf.predict(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED — Confidence: {prob*100:.1f}%")
    else:
        st.success(f"✅ LEGIT TRANSACTION — Fraud Probability: {prob*100:.1f}%")

    st.markdown("#### Fraud Probability")
    st.progress(float(prob))
    st.caption(f"{prob*100:.2f}% chance of fraud")

