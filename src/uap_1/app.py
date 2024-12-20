# app.py
import streamlit as st
from input_and_predict import input_and_predict
from shap_visualization import shap_visualization
from model_comparison import model_comparison

def main():
    st.title("AI Prediction Web App")

    # Sidebar untuk memilih model dan input data
    model_choice = st.sidebar.selectbox("Choose Model", ("TabNet", "XGBoost"))
    
    input_data = st.text_area("Enter Input Data (comma-separated)")
    
    # Memilih fitur
    feature_choice = st.sidebar.radio("Choose a feature", ("Prediction", "SHAP Explanation", "Model Comparison"))
    
    if feature_choice == "Prediction":
        input_and_predict()
    elif feature_choice == "SHAP Explanation":
        shap_visualization(model_choice, input_data)
    elif feature_choice == "Model Comparison":
        model_comparison()

if __name__ == "__main__":
    main()
