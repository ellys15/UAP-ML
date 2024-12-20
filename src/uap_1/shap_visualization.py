# shap_visualization.py
import shap
import streamlit as st
from utils import load_xgb_model, load_tabnet_model
from sklearn.preprocessing import StandardScaler

def shap_visualization(model_choice, input_data):
    if input_data:
        try:
            # Persiapkan data
            input_array = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)

            # Memuat model
            tabnet_model = load_tabnet_model()
            xgb_model = load_xgb_model()

            # Skala data
            scaler = StandardScaler()
            input_array_scaled = scaler.fit_transform(input_array)

            if model_choice == 'TabNet':
                # SHAP untuk TabNet
                explainer_tabnet = shap.Explainer(tabnet_model)
                shap_values_tabnet = explainer_tabnet(input_array_scaled)
                st.write("TabNet SHAP Explanation:")
                shap.summary_plot(shap_values_tabnet, input_array_scaled)

            elif model_choice == 'XGBoost':
                # SHAP untuk XGBoost
                explainer_xgb = shap.Explainer(xgb_model)
                shap_values_xgb = explainer_xgb(input_array)
                st.write("XGBoost SHAP Explanation:")
                shap.summary_plot(shap_values_xgb, input_array)

        except ValueError:
            st.error("Invalid input format. Please enter comma-separated numeric values.")
    else:
        st.error("Please provide input data for SHAP explanation.")

