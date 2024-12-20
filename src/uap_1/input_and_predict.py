# input_and_predict.py
import numpy as np
import streamlit as st
from utils import load_tokenizer, load_xgb_model, load_tabnet_model
from sklearn.preprocessing import StandardScaler

def input_and_predict():
    # Memilih model di sidebar
    model_choice = st.sidebar.selectbox("Choose Model", ("TabNet", "XGBoost"))

    # Input data
    st.title("AI Prediction Web App")
    st.write("Enter your data below for prediction:")

    input_data = st.text_area("Enter Input Data (comma-separated)")

    if st.button("Make Prediction"):
        if input_data:
            try:
                # Persiapkan data input dari pengguna
                input_array = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)

                # Memuat model dan tokenizer
                tabnet_model = load_tabnet_model()
                xgb_model = load_xgb_model()
                tokenizer = load_tokenizer()

                # Skala data
                scaler = StandardScaler()
                input_array_scaled = scaler.fit_transform(input_array)

                if model_choice == "TabNet":
                    tabnet_pred = tabnet_model.predict(input_array_scaled)
                    st.write(f"Prediction from TabNet: {tabnet_pred}")
                elif model_choice == "XGBoost":
                    dmatrix = xgb.DMatrix(input_array)
                    xgb_pred = xgb_model.predict(dmatrix)
                    st.write(f"Prediction from XGBoost: {xgb_pred}")

            except ValueError:
                st.error("Please enter valid comma-separated numerical values.")
        else:
            st.error("Please provide input data for prediction.")
