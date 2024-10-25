# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Function to test the stationarity of a time series
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(24).mean()  # Rolling mean
    rolstd = timeseries.rolling(24).std()    # Rolling std deviation
    
    # Plot rolling statistics
    plt.figure(figsize=(10, 6))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    st.pyplot(plt)  # Use Streamlit's pyplot

    # Perform the Dickey-Fuller test
    st.write('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    st.write(dfoutput)

# Streamlit app layout
st.title('ARIMA Time Series Forecasting')

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load the dataset
    train = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Data Preview:")
    st.write(train.head())

    # Assuming 'Count' is the target variable for ARIMA
    if 'Count' in train.columns:
        # Log transformation of the 'Count' column
        train_log = np.log(train['Count'])

        # Test for stationarity
        test_stationarity(train_log)

        # Define and fit the ARIMA model using auto_arima
        model = auto_arima(train_log,
                           seasonal=False,
                           m=1,
                           start_p=0,
                           start_d=0,
                           start_q=0,
                           max_p=3,
                           max_d=2,
                           max_q=3,
                           trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

        # Print the summary of the best ARIMA model
        st.write("Best ARIMA Model Summary:")
        st.text(model.summary())

        # Generate forecasts
        forecast_steps = 30  # Forecasting 30 steps ahead
        model_fit = model.fit(train_log)
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=train.index[-1], periods=forecast_steps + 1, freq='D')[1:]

        # Get forecast values and confidence intervals
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Create a DataFrame for the forecast
        forecast_df = pd.DataFrame({
            'Forecast': forecast_values,
            'Lower CI': conf_int.iloc[:, 0],
            'Upper CI': conf_int.iloc[:, 1]
        }, index=forecast_index)

        # Plot historical data and forecast
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train['Count'], label='Historical Data')
        plt.plot(forecast_df.index, forecast_df['Forecast'], color='red', label='Forecast')
        plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='pink', alpha=0.3)
        plt.title('ARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend(loc='best')
        st.pyplot(plt)  # Use Streamlit's pyplot
    else:
        st.error("The uploaded CSV file must contain a 'Count' column.")
