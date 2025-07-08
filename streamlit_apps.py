import streamlit as st
import requests
import json

st.set_page_config(page_title="CLV Prediction", layout="centered")

st.title("Customer Lifetime Value (CLV) Prediction")
st.markdown("Enter customer details to predict the CLV.")

total_purchases = st.number_input("Total Purchases ($)", min_value=0.0, format="%.2f")
engagement_score = st.number_input("Engagement Score", min_value=0.0, format="%.2f")

if st.button("Predict CLV"):
    input_data = {"total_purchases": total_purchases, "engagement_score": engagement_score}
    
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)

        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                st.success(f"Predicted CLV: ${result['prediction']:.2f}")
            else:
                st.error(f"Error: {result.get('error', 'Unexpected error occurred')}")
        else:
            st.error(f"Server error: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend. Ensure Flask is running.")

    except requests.exceptions.JSONDecodeError:
        st.error("Invalid response from backend. Check Flask logs.")
