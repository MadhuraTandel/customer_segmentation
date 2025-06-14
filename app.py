import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved objects
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('features.pkl')

st.title("Customer Segmentation App")
st.write("Enter customer details to predict their segment (cluster).")

# Input form
with st.form("customer_form"):
    income = st.number_input("Income", min_value=0)
    age = st.number_input("Age", min_value=0, max_value=100)
    education = st.selectbox("Education", ['Graduation', 'PhD', 'Master', 'Basic', '2n Cycle'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'])
    kidhome = st.number_input("Kids at Home", min_value=0)
    teenhome = st.number_input("Teens at Home", min_value=0)
    recency = st.number_input("Recency", min_value=0)

    # Add other features as needed...

    submitted = st.form_submit_button("Predict Segment")

if submitted:
    # Prepare input
    input_dict = {
        'Income': income,
        'Age': age,
        'Kidhome': kidhome,
        'Teenhome': teenhome,
        'Recency': recency,
        f'Education_{education}': 1,
        f'Marital_Status_{marital_status}': 1,
    }

    # Create full input feature vector
    input_df = pd.DataFrame([input_dict])
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    cluster = model.predict(input_scaled)[0]

    st.success(f"The customer belongs to **Cluster {cluster}**")
