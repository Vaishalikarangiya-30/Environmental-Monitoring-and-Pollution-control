# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("🌿 Air Pollution Monitoring & Prediction Dashboard")

# Get current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
try:
    model = joblib.load(os.path.join(BASE_DIR, 'pollution_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Run 'train_and_save_model.py' first!")
    st.stop()

# Upload Excel file
data_file = st.file_uploader("Upload Air Quality Excel", type=["xlsx"])

if data_file:
    try:
        df = pd.read_excel(data_file)
        df = df.iloc[:, :-2]  # Remove empty columns

        # Show preview
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        # Replace -200 with NaN and fill missing numeric values
        df.replace(-200, np.nan, inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Features
        features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)']

        # Check if all features exist
        if not all(f in df.columns for f in features):
            st.error("❌ Uploaded file is missing required features.")
            st.stop()

        # Scale features
        X_input = scaler.transform(df[features])

        # Predict
        if st.button("Predict C6H6(GT) Levels"):
            predictions = model.predict(X_input)
            df['Predicted_C6H6'] = predictions

            st.subheader("Prediction Results")
            st.dataframe(df[features + ['Predicted_C6H6']])

            # Warning for high pollution
            threshold = 50
            if np.max(predictions) > threshold:
                st.warning(f"⚠️ High Benzene (C6H6) Level Detected! Max: {np.max(predictions):.2f}")

            # Line chart
            st.subheader("Predicted C6H6(GT) Levels Chart")
            st.line_chart(df['Predicted_C6H6'])

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
