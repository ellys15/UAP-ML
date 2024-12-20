# model_comparison.py
import streamlit as st
import numpy as np

def model_comparison():
    st.subheader("Model Performance Comparison")

    # Metrik evaluasi untuk TabNet dan XGBoost
    mae_tabnet = [226119.32, 1213711.50, 1724347.20]  # Ganti dengan nilai sesungguhnya
    mae_xgb = [17018.26, 71463.30, 106231.47]  # Ganti dengan nilai sesungguhnya
    rmse_tabnet = [944921.26, 4554055.07, 6974132.80]  # Ganti dengan nilai sesungguhnya
    rmse_xgb = [47825.10, 190554.48, 451952.57]  # Ganti dengan nilai sesungguhnya
    r2_tabnet = [-0.047, 0.027, 0.045]  # Ganti dengan nilai sesungguhnya
    r2_xgb = [0.997, 0.998, 0.995]  # Ganti dengan nilai sesungguhnya

    st.write(f"MAE per target: TabNet = {mae_tabnet}, XGBoost = {mae_xgb}")
    st.write(f"RMSE per target: TabNet = {rmse_tabnet}, XGBoost = {rmse_xgb}")
    st.write(f"R² per target: TabNet = {r2_tabnet}, XGBoost = {r2_xgb}")

    # Visualisasi grafis untuk membandingkan
    metrics = ['MAE', 'RMSE', 'R²']
    tabnet_metrics = [mae_tabnet, rmse_tabnet, r2_tabnet]
    xgb_metrics = [mae_xgb, rmse_xgb, r2_xgb]

    # Plot bar chart untuk perbandingan
    st.bar_chart({
        'TabNet': tabnet_metrics,
        'XGBoost': xgb_metrics
    })
