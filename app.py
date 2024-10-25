# app.py

import os
os.system("pip uninstall -y pmdarima")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import io

st.title("ARIMA Time Series Forecasting")

# File uploader for sample CSV
uploaded_file = st.file_uploader("Upload a CSV file with 'Month' and 'Sales Amt' columns", type="csv")

# Forecast period selection slider
forecast_period = st.slider("Select number of months to forecast", min_value=1, max_value=24, value=6)

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Check if required columns are present
    if 'Month' in data.columns and 'Sales Amt' in data.columns:
        # Parse dates and set index
        data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')
        data.set_index('Month', inplace=True)
        
        st.write("Uploaded Data:")
        st.write(data)
        
        # Plot original data
        st.subheader("Sales Amount Over Time")
        plt.figure(figsize=(10, 4))
        plt.plot(data['Sales Amt'], label='Sales Amt', color='blue')
        plt.title("Sales Amount Over Time")
        plt.xlabel("Month")
        plt.ylabel("Sales Amount")
        plt.legend()
        st.pyplot(plt)

        # Train ARIMA model
        model = ARIMA(data['Sales Amt'], order=(2, 1, 0))
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=forecast_period)
        forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_period, freq='MS')
        forecast_series = pd.Series(forecast, index=forecast_index)

        # Plot forecast
        st.subheader("Forecasted Sales Amount")
        plt.figure(figsize=(10, 4))
        plt.plot(data['Sales Amt'], label='Sales Amt', color='blue')
        plt.plot(forecast_series, label='Forecast', color='red', linestyle='--')
        plt.title("Sales Amount with Forecast")
        plt.xlabel("Month")
        plt.ylabel("Sales Amount")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("The uploaded file does not contain the required 'Month' and 'Sales Amt' columns.")

# Footer
st.write("Developed by Dillip Meher")
