import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# Title for the Streamlit app
st.title("Sales Forecasting using Auto ARIMA")

# File upload option
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the data
    data = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("Uploaded Data:", data)

    # Ensure the 'Month' and 'Sales Amt' columns exist in the file
    if 'Month' in data.columns and 'Sales Amt' in data.columns:
        # Parse the dates and set as index
        data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')
        data.set_index('Month', inplace=True)

        # Display time series plot of the sales data
        st.subheader("Sales Data Over Time")
        st.line_chart(data['Sales Amt'])

        # Model training with Auto ARIMA
        st.subheader("Auto ARIMA Model Training")
        with st.spinner("Training Auto ARIMA model..."):
            model = auto_arima(data['Sales Amt'], start_p=1, start_q=1,
                               max_p=5, max_q=5, seasonal=False,
                               stepwise=True, suppress_warnings=True, trace=True)

        st.success("Model training complete!")

        # Display the summary of the model
        st.subheader("Model Summary")
        st.text(model.summary())

        # Forecast the future sales
        n_periods = st.slider("Select number of periods to forecast", min_value=1, max_value=12, value=2)
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

        # Create a forecast DataFrame for better visualization
        future_dates = pd.date_range(data.index[-1], periods=n_periods+1, freq='M')[1:]
        forecast_df = pd.DataFrame({
            'Forecast': forecast,
            'Lower Confidence Interval': conf_int[:, 0],
            'Upper Confidence Interval': conf_int[:, 1]
        }, index=future_dates)

        st.subheader("Forecasted Sales")
        st.write(forecast_df)

        # Plot the forecast along with the historical data
        st.subheader("Sales Forecast Plot")
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Sales Amt'], label="Historical Sales")
        plt.plot(forecast_df.index, forecast_df['Forecast'], label="Forecasted Sales", color='green')
        plt.fill_between(forecast_df.index, forecast_df['Lower Confidence Interval'], forecast_df['Upper Confidence Interval'],
                         color='gray', alpha=0.2, label='Confidence Interval')
        plt.legend()
        plt.title("Sales Forecast using Auto ARIMA")
        plt.xlabel("Date")
        plt.ylabel("Sales Amount")
        st.pyplot(plt.gcf())

    else:
        st.error("Please ensure your CSV file has 'Month' and 'Sales Amt' columns.")
else:
    st.info("Please upload a CSV file to start.")

