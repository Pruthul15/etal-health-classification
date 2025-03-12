import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load models
log_reg = pickle.load(open("logistic_regression_model.pkl", "rb"))
rf_clf = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.title("Fetal Health Classification")
st.write("Upload new data and classify fetal health.")

# User input form
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    scaled_data = scaler.transform(data)

    # Model Predictions
    log_pred = log_reg.predict(scaled_data)
    rf_pred = rf_clf.predict(scaled_data)

    # Display results
    st.subheader("Logistic Regression Predictions:")
    st.write(log_pred)

    st.subheader("Random Forest Predictions:")
    st.write(rf_pred)
