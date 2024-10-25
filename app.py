import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Function to create sample data
def create_sample_data():
    sample_data = {
        'Month': ['Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 
                  'Jun-24', 'Jul-24', 'Aug-24', 'Sep-24'],
        'Sales Amt': [9356, 4891, 5824, 8116, 2864, 2326, 2240, 6717, 2864]
    }
    return pd.DataFrame(sample_data)

# Streamlit app
st.title("Time Series Forecasting with ARIMA")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Load sample data if no file is uploaded
if uploaded_file is None:
    data = create_sample_data()
    st.write("Using sample data:")
else:
    data = pd.read_csv(uploaded_file)

# Display the uploaded data
st.write("Data:")
st.dataframe(data)

# Ensure the 'Month' column is in the correct format
data['Month'] = pd.to_datetime(data['Month'], format='%b-%y', errors='coerce')
data['Sales Amt'] = pd.to_numeric(data['Sales Amt'], errors='coerce')

# Check for invalid dates
if data['Month'].isnull().any():
    st.error("Some 'Month' values are invalid and could not be parsed. Please ensure dates are in 'MMM-YY' format (e.g., Jan-24).")
else:
    # Set 'Month' as the index
    data.set_index('Month', inplace=True)

    # ARIMA model
    model = auto_arima(data['Sales Amt'], seasonal=False, stepwise=True)
    
    # Forecasting
    forecast_period = st.slider("Select number of months to forecast:", 1, 12, 3)
    forecast, conf_int = model.predict(n_periods=forecast_period, return_conf_int=True)

    # Create forecast index
    forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(months=1), 
                                    periods=forecast_period, freq='MS')

    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecasted Sales'])

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Sales Amt'], label='Historical Sales', marker='o')
    plt.plot(forecast_df.index, forecast_df['Forecasted Sales'], label='Forecasted Sales', marker='o', color='orange')
    plt.fill_between(forecast_df.index, conf_int[:, 0], conf_int[:, 1], color='orange', alpha=0.2, label='Confidence Interval')
    plt.title('Sales Forecasting')
    plt.xlabel('Month')
    plt.ylabel('Sales Amount')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # Display forecast values
    st.write("Forecasted Sales Values:")
    st.dataframe(forecast_df)
