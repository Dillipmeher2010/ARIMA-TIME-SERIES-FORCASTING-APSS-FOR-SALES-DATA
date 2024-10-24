import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import os

# Create a sample DataFrame for the Excel file
sample_data = {
    'Month': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', 
              '2024-06-01', '2024-07-01', '2024-08-01'],
    'Sales Amt': [5319, 9990, 7597, 8485, 5243, 5367, 3168, 5202]
}
sample_df = pd.DataFrame(sample_data)

# Save the sample DataFrame to an Excel file
sample_file_path = 'sample_sales_data.xlsx'
sample_df.to_excel(sample_file_path, index=False)

# Streamlit app layout
st.title("Sales Forecasting App with Auto ARIMA")
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
        df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m-%d', errors='coerce')  # Ensure correct format
        df.dropna(subset=['Month', 'Sales Amt'], inplace=True)  # Drop rows with NaT values

        # Set the Month as the index
        df.set_index('Month', inplace=True)

        # Display the cleaned DataFrame for debugging
        st.write("Cleaned DataFrame:", df)

        # Fit the Auto ARIMA model
        st.subheader("Training Auto ARIMA Model")
        with st.spinner("Training..."):
            try:
                model = auto_arima(df['Sales Amt'], seasonal=False, stepwise=True, trace=True)
                st.success("Model training complete!")
            except Exception as e:
                st.error(f"Error in model fitting: {e}")
                st.stop()

        # Forecasting
        n_periods = st.slider("Select the number of periods to forecast:", min_value=1, max_value=12, value=3)
        forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

        # Create a DataFrame for the forecast
        forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='M')
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
        conf_int_df = pd.DataFrame(conf_int, index=forecast_index, columns=['Lower Bound', 'Upper Bound'])

        # Display the forecasting results
        st.write("Forecasting Results:")
        st.write(forecast_df)

        # Plot the results
        st.subheader("Forecast Plot")
        plt.figure(figsize=(10, 5))
        plt.plot(df['Sales Amt'], label='Historical Sales', color='blue')
        plt.plot(forecast_df['Forecast'], label='Forecast', color='orange')
        plt.fill_between(forecast_index, conf_int_df['Lower Bound'], conf_int_df['Upper Bound'], color='lightgrey', alpha=0.5)
        plt.title('Sales Forecast with Auto ARIMA')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount')
        plt.legend()
        st.pyplot(plt)

        # Save the forecasting results to a new Excel file for download
        forecast_file = "forecasted_sales_data.xlsx"
        combined_forecast = pd.concat([forecast_df, conf_int_df], axis=1)
        combined_forecast.to_excel(forecast_file, index=True)

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
