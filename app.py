import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import io

st.title("Time Series Forecasting with Auto ARIMA")

# Sample data for download
sample_data = pd.DataFrame({
    "Month": ["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24"],
    "Sales Amt": [9356, 4891, 5824, 8116, 2864, 2326, 2240, 6717, 2864]
})

# Button to download the sample file
st.sidebar.header("Sample File")
sample_file = io.BytesIO()
sample_data.to_csv(sample_file, index=False)
sample_file.seek(0)
st.sidebar.download_button(
    label="Download Sample CSV",
    data=sample_file,
    file_name="sample_sales_data.csv",
    mime="text/csv"
)

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

        # Fit Auto ARIMA model
        try:
            model = auto_arima(data['Sales Amt'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
            st.success("Auto ARIMA model successfully fitted with order: {}".format(model.order))

            # Forecasting
            forecast = model.predict(n_periods=forecast_period)
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
            st.error(f"An error occurred while fitting the Auto ARIMA model: {e}")
