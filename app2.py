import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.title("Retail Customer Segmentation")
st.write("Enter RFM details to predict customer cluster.")

# User Inputs
recency = st.number_input("Recency (days since last purchase)", 0, 365, 30)
frequency = st.number_input("Frequency (number of purchases)", 1, 1000, 50)
monetary = st.number_input("Monetary (total spend)", 0.0, 10000.0, 500.0)

# Preprocess and Predict
if st.button("Predict Cluster"):
    input_data = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    st.subheader("Predicted Cluster:")
    st.success(f"Cluster {cluster}")
    # Optional: Describe based on your analysis
    if cluster == 0:
        st.info("Low-value customer (high recency, low frequency/spend).")
    # Add for other clusters