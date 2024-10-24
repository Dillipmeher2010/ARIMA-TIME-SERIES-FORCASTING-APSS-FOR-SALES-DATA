import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# Set title of the app
st.title("Time Series Forecasting with Auto ARIMA")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("Uploaded Data:", data)

    # Ensure 'Month' and 'Sales Amt' columns are present
    if 'Month' in data.columns and 'Sales Amt' in data.columns:
        # Parse dates
        data['Month'] = pd.to_datetime(data['Month'], format='%b-%y')
        data.set_index('Month', inplace=True)

        # Plotting the time series
        st.subheader("Time Series Data")
        st.line_chart(data['Sales Amt'])

        # Train Auto ARIMA model
        st.subheader("Training Auto ARIMA Model")
        with st.spinner("Training..."):
            model = auto_arima(data['Sales Amt'], seasonal=False, stepwise=True, trace=True)

        st.success("Model training complete!")

        # Display model summary
        st.subheader("Model Summary")
        st.text(model.summary())

        # Forecasting
        n_periods = st.slider("Select the number of periods to forecast:", min_value=1, max_value=12, value=3)
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

        # Create a DataFrame for the forecast
        forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='M')
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
        conf_int_df = pd.DataFrame(conf_int, index=forecast_index, columns=['Lower Bound', 'Upper Bound'])

        # Display forecast data
        st.subheader("Forecast Results")
        st.write(forecast_df.join(conf_int_df))

        # Plotting forecast
        st.subheader("Forecast Plot")
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Sales Amt'], label='Historical Sales', color='blue')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green')
        plt.fill_between(conf_int_df.index, conf_int_df['Lower Bound'], conf_int_df['Upper Bound'], color='gray', alpha=0.5)
        plt.title("Sales Forecast with Auto ARIMA")
        plt.xlabel("Month")
        plt.ylabel("Sales Amount")
        plt.legend()
        st.pyplot(plt.gcf())
    else:
        st.error("Ensure your CSV has 'Month' and 'Sales Amt' columns.")
else:
    st.info("Upload a CSV file to start.")
