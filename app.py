import streamlit as st
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Auto ARIMA Time Series Forecasting App")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the CSV/XLSX file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(df)

    # Check if data has the required columns
    if 'Month' in df.columns and 'Sales Amt' in df.columns:
        # Convert 'Month' to datetime
        df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
        df.set_index('Month', inplace=True)

        # Handle missing values (optional: interpolation)
        df['Sales Amt'].interpolate(method='linear', inplace=True)

        # Display cleaned data
        st.write("Cleaned Data:")
        st.write(df)

        # Model Training using Auto ARIMA
        st.write("Fitting Auto ARIMA model...")
        model = auto_arima(df['Sales Amt'], seasonal=True, trace=True, suppress_warnings=True)

        # Display ARIMA order
        st.write(f"Best ARIMA model order: {model.order}, Seasonal order: {model.seasonal_order}")

        # Forecast for the next 2 months (Sep-24, Oct-24)
        forecast_periods = 2
        forecast = model.predict(n_periods=forecast_periods)

        # Creating forecast dates for display
        last_date = df.index[-1]
        future_dates = pd.date_range(last_date, periods=forecast_periods + 1, freq='M')[1:]

        # Create a DataFrame for forecast results
        forecast_df = pd.DataFrame({
            'Month': future_dates,
            'Forecasted Sales': forecast
        })
        forecast_df.set_index('Month', inplace=True)

        # Display forecast results
        st.write("Forecast for next months:")
        st.write(forecast_df)

        # Plot the actual and forecasted values
        st.write("Sales Forecast Plot:")
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Sales Amt'], label="Actual Sales")
        plt.plot(forecast_df.index, forecast_df['Forecasted Sales'], label="Forecasted Sales", linestyle='--')
        plt.xlabel("Month")
        plt.ylabel("Sales Amount")
        plt.title("Sales Forecast using Auto ARIMA")
        plt.legend()
        st.pyplot(plt)

    else:
        st.error("The uploaded file does not contain the required columns 'Month' and 'Sales Amt'. Please check the file format.")

else:
    st.info("Please upload a CSV or Excel file to get started.")
