import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Title
st.title("ARIMA Time Series Forecasting App")

# File upload
uploaded_file = st.file_uploader("Upload CSV file with 'Month' and 'Sales Amt' columns", type="csv")

# Ensure a file is uploaded
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Validate data columns
    if data.empty or 'Month' not in data.columns or 'Sales Amt' not in data.columns:
        st.error("The uploaded file is either empty or not in the expected format. Please upload a CSV file with 'Month' and 'Sales Amt' columns.")
        st.stop()
    
    # Convert 'Month' to datetime format
    data['Month'] = pd.to_datetime(data['Month'], format='%b-%y', errors='coerce')
    data.set_index('Month', inplace=True)
    
    # Check for invalid dates
    if data.index.hasnans:
        st.error("The 'Month' column contains invalid date values. Ensure dates are in 'MMM-YY' format (e.g., Jan-24).")
        st.stop()

    # Forecasting setup
    st.write("Data preview:", data.head())

    # Input for forecast period
    forecast_period = st.slider("Select months to forecast", min_value=1, max_value=24, value=6)

    # ARIMA Model
    model = ARIMA(data['Sales Amt'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_period)

    # Extend the index for the forecast period
    forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_period, freq='MS')
    forecast_series = pd.Series(forecast, index=forecast_index)

    # Plotting
    fig, ax = plt.subplots()
    data['Sales Amt'].plot(ax=ax, label='Historical Data')
    forecast_series.plot(ax=ax, label='Forecast', color='red')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Sales Amt")
    plt.title("Sales Amt Forecast")

    st.pyplot(fig)
