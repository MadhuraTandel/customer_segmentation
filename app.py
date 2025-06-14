import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and preprocessing objects
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('features.pkl')  # list of features used during training

st.title("Customer Segmentation App")
st.write("Fill in the customer details to predict the segment (cluster) using KMeans clustering.")

# Streamlit input form
with st.form("input_form"):
    income = st.number_input("Income", min_value=0)
    age = st.number_input("Age", min_value=18, max_value=100)
    kidhome = st.number_input("Number of Kids at Home", min_value=0)
    teenhome = st.number_input("Number of Teens at Home", min_value=0)
    recency = st.number_input("Recency (days since last purchase)", min_value=0)

    education = st.selectbox("Education", ['Graduation', 'PhD', 'Master', 'Basic', '2n Cycle'])
    marital_status = st.selectbox("Marital Status", ['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'])

    submit_button = st.form_submit_button("Predict Segment")

if submit_button:
    # 1. Create input dictionary
    input_dict = {
        'Income': income,
        'Age': age,
        'Kidhome': kidhome,
        'Teenhome': teenhome,
        'Recency': recency,
        f'Education_{education}': 1,
        f'Marital_Status_{marital_status}': 1,
    }

    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # 3. Add missing features with default value 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # 4. Ensure correct column order and remove any extras
    input_df = input_df[feature_names]

    # 5. Check for NaNs and fill (if any)
    input_df.fillna(0, inplace=True)

    # 6. Scale input and predict
    input_scaled = scaler.transform(input_df.to_numpy())
    cluster = model.predict(input_scaled)[0]

    # 7. Display result
    st.success(f"The customer belongs to **Cluster {cluster}**.")
