import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import base64

# Function to generate a download link for the forecasted data
def download_link(data, filename):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Forecasted Data</a>'
    return href

# Streamlit app title
st.title("Time Series Forecasting with Auto ARIMA")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write("Data Preview:")
    st.dataframe(df)

    # Assuming the CSV has a column named 'Sales Amt' for the forecasting
    if 'Sales Amt' in df.columns:
        # Prepare data for modeling
        df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='M')  # Adjust date if necessary
        df.set_index('Date', inplace=True)
        y = df['Sales Amt']

        # Fit Auto ARIMA model
        with st.spinner("Fitting Auto ARIMA model..."):
            model = auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True)

        # Forecasting
        n_periods = st.number_input("Number of periods to forecast", min_value=1, max_value=24, value=12)
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

        # Create a DataFrame for the forecast
        forecast_index = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='M')
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=["Forecast"])
        forecast_df['Lower CI'] = conf_int[:, 0]
        forecast_df['Upper CI'] = conf_int[:, 1]

        # Display forecast results
        st.write("Forecast Results:")
        st.dataframe(forecast_df)

        # Plotting the results
        plt.figure(figsize=(10, 5))
        plt.plot(y, label='Historical Data', color='blue')
        plt.plot(forecast_df['Forecast'], label='Forecast', color='orange')
        plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='lightgray', alpha=0.5)
        plt.title('Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # Provide download link for forecasted data
        st.markdown(download_link(forecast_df, "forecasted_sales.csv"), unsafe_allow_html=True)

    else:
        st.error("The uploaded file must contain a 'Sales Amt' column for forecasting.")
