import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from io import BytesIO

# Title
st.title("Sales Forecasting using Auto ARIMA")

# Sample file download
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# Sample data
sample_data = pd.DataFrame({
    'Month': ['Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24', 'Jul-24', 'Aug-24'],
    'Sales Amt': [5319, 9990, 7597, 8485, 5243, 5367, 3168, 5202]
})

st.markdown("### Download the sample file:")
st.download_button(label="Download Sample Data", data=to_excel(sample_data), file_name='sample_sales_data.xlsx')

# File upload section
st.markdown("### Upload your Excel file:")
uploaded_file = st.file_uploader("Upload your sales data", type=["xlsx"])

# Process uploaded file
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Check the uploaded file format
    if 'Month' not in df.columns or 'Sales Amt' not in df.columns:
        st.error("The file must contain 'Month' and 'Sales Amt' columns.")
    else:
        # Convert 'Month' to datetime format
        df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
        df.set_index('Month', inplace=True)
        
        # Train the Auto ARIMA model
        st.markdown("### Auto ARIMA Model Forecast")
        
        # Auto ARIMA fitting
        model = auto_arima(df['Sales Amt'], seasonal=False, trace=True)
        model.fit(df['Sales Amt'])
        
        # Forecast the next two months
        forecast = model.predict(n_periods=2)
        forecast_df = pd.DataFrame({
            'Month': pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=2, freq='M'),
            'Forecast Sales Amt': forecast
        })
        
        # Show the forecasted results
        st.write(forecast_df)
        
        # Provide option to download forecasted results
        forecast_download = to_excel(forecast_df)
        st.download_button(label="Download Forecast", data=forecast_download, file_name='forecast_sales_data.xlsx')
