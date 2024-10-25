import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("Time Series Forecasting with ARIMA")

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload a CSV file with 'Month' and 'Sales Amt' columns", type="csv")

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Check if the required columns are in the data
    if 'Month' not in data.columns or 'Sales Amt' not in data.columns:
        st.error("Uploaded file must contain 'Month' and 'Sales Amt' columns.")
    else:
        # Clean and convert 'Month' column to datetime
        data['Month'] = data['Month'].str.strip()  # Remove extra spaces
        data['Month'] = pd.to_datetime(data['Month'], format='%b-%y', errors='coerce')

        # Check for invalid date rows
        invalid_dates = data[data['Month'].isna()]
        if not invalid_dates.empty:
            st.error("Some 'Month' values are invalid and could not be parsed. Ensure dates are in 'MMM-YY' format (e.g., Jan-24).")
            st.write("Invalid rows:")
            st.write(invalid_dates)
            st.stop()

        # Set 'Month' as index and ensure 'Sales Amt' is numeric
        data.set_index('Month', inplace=True)
        data['Sales Amt'] = pd.to_numeric(data['Sales Amt'], errors='coerce')

        # Drop rows with NaN values in 'Sales Amt'
        data = data.dropna()

        # Plot original data
        st.subheader("Original Sales Data")
        st.line_chart(data['Sales Amt'])

        # Forecasting parameters
        forecast_period = st.slider("Select the number of months for forecasting", min_value=1, max_value=24, value=12)

        # Fit ARIMA model
        try:
            model = ARIMA(data['Sales Amt'], order=(1, 1, 1))
            model_fit = model.fit()
            st.success("ARIMA model successfully fitted.")

            # Forecasting
            forecast = model_fit.forecast(steps=forecast_period)
            forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_period, freq='MS')
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

            # Plot actual and forecasted data
            st.subheader("Sales Forecast")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Sales Amt'], label="Actual Sales")
            plt.plot(forecast_df, label="Forecasted Sales", color="orange")
            plt.xlabel("Month")
            plt.ylabel("Sales Amt")
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error(f"An error occurred while fitting the ARIMA model: {e}")
