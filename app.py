import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Streamlit app layout
st.title("Sales Forecasting App with Auto ARIMA")
st.write("Upload your sales data in the format of the sample file below:")

# File upload section
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)

    # Display the DataFrame for debugging
    st.write("Uploaded DataFrame:", df)

    # Check for required columns
    if 'Month' in df.columns and 'Sales Amt' in df.columns:
        # Prepare the data
        df['Month'] = pd.to_datetime(df['Month'], format='%b-%y', errors='coerce')
        df.set_index('Month', inplace=True)

        # Drop rows with NaT values if date conversion failed
        df.dropna(inplace=True)
        st.write("DataFrame after date conversion:", df)

        # Check if there are enough rows to fit the model
        if len(df) < 2:
            st.error("The DataFrame must contain at least 2 valid rows for fitting the model.")
        else:
            # Fit the Auto ARIMA model
            st.subheader("Training Auto ARIMA Model")
            try:
                model = auto_arima(df['Sales Amt'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
                st.success("Model training complete!")
            except Exception as e:
                st.error(f"Model fitting failed: {e}")
                st.stop()  # Stop further execution if fitting fails

            # Create a future DataFrame for forecasting
            n_periods = st.slider("Select the number of periods to forecast:", min_value=1, max_value=12, value=3)
            try:
                # Forecast the future sales
                forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
                forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='M')
                forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

                # Display the results
                st.write("Forecasting Results:")
                st.write(forecast_df)

                # Plot the results
                st.subheader("Forecast Plot")
                plt.figure(figsize=(10, 6))
                plt.plot(df.index, df['Sales Amt'], label='Historical Sales', color='blue', marker='o')
                plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green', marker='o', linestyle='--')
                plt.fill_between(forecast_df.index, conf_int[:, 0], conf_int[:, 1], color='lightgreen', alpha=0.3)
                plt.title("Sales Forecast with Auto ARIMA")
                plt.xlabel("Month")
                plt.ylabel("Sales Amount")
                plt.xticks(rotation=45)
                plt.axvline(x=df.index[-1], color='red', linestyle='--', label='Forecast Start')  # Line indicating forecast start
                plt.legend()
                plt.grid()
                st.pyplot(plt)

            except Exception as e:
                st.error(f"An error occurred during forecasting: {e}")
    else:
        st.error("Uploaded file must contain 'Month' and 'Sales Amt' columns.")
