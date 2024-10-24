import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os

# Sample DataFrame creation
sample_data = {
    'Month': ['Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 
              'Jun-24', 'Jul-24', 'Aug-24'],
    'Sales Amt': [5319, 9990, 7597, 8485, 5243, 5367, 3168, 5202]
}
sample_df = pd.DataFrame(sample_data)

# Save the sample DataFrame to an Excel file
sample_file_path = 'sample_sales_data.xlsx'
sample_df.to_excel(sample_file_path, index=False)

# Streamlit app layout
st.title("Sales Forecasting App with ARIMA")
st.write("Upload your sales data in the format of the sample file below:")

# Display download link for the sample file
with open(sample_file_path, "rb") as f:
    st.download_button(
        label="Download Sample Sales Data",
        data=f,
        file_name=sample_file_path,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

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
            # Fit the ARIMA model
            st.subheader("Training ARIMA Model")
            order = (1, 1, 1)  # Example order (p, d, q); you can adjust this
            with st.spinner("Training..."):
                model = ARIMA(df['Sales Amt'], order=order)
                model_fit = model.fit()
                st.success("Model training complete!")

            # Create a future DataFrame for forecasting
            n_periods = st.slider("Select the number of periods to forecast:", min_value=1, max_value=12, value=3)
            forecast = model_fit.forecast(steps=n_periods)
            forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='M')
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])

            # Check the forecast DataFrame
            st.write("Forecast DataFrame:", forecast_df)

            # Display the results
            st.write("Forecasting Results:")
            st.write(forecast_df)

            # Plot the results
            st.subheader("Forecast Plot")
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['Sales Amt'], label='Historical Sales', color='blue')
            plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green')
            plt.title("Sales Forecast with ARIMA")
            plt.xlabel("Month")
            plt.ylabel("Sales Amount")
            plt.legend()
            st.pyplot(plt.gcf())

            # Save the forecasting results to a new Excel file for download
            forecast_file = "forecasted_sales_data.xlsx"
            forecast_df.to_excel(forecast_file)

            # Create a download button for the forecasting results
            with open(forecast_file, "rb") as f:
                st.download_button(
                    label="Download Forecasting Data",
                    data=f,
                    file_name=forecast_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.error("Uploaded file must contain 'Month' and 'Sales Amt' columns.")
