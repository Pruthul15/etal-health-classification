import streamlit as st
import pandas as pd
import joblib

# Load models
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Fetal Health Classification")
st.write("Upload data to predict fetal health")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())
    df_scaled = scaler.transform(df)
    predictions = rf_model.predict(df_scaled)
    st.write("Predictions:", predictions)
