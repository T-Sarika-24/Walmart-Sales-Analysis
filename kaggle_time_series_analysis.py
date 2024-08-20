
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the dataset
# Assuming the dataset is downloaded from Kaggle and stored as 'dataset.csv'
data = pd.read_csv('dataset.csv')

# Data Cleaning
data.dropna(inplace=True)

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Correlation analysis
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Plotting the time series data
plt.figure(figsize=(10,6))
plt.plot(data['Target'], label='Target')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.show()

# Naive Approach
data['Naive_Forecast'] = data['Target'].shift(1)

# Calculate MAE for Naive Approach
naive_mae = np.mean(np.abs(data['Target'] - data['Naive_Forecast']))
print(f'Naive MAE: {naive_mae}')

# Moving Average Model
window_size = 3
data['Moving_Average_Forecast'] = data['Target'].rolling(window=window_size).mean()

# Calculate MAE for Moving Average
ma_mae = np.mean(np.abs(data['Target'][window_size:] - data['Moving_Average_Forecast'][window_size:]))
print(f'Moving Average MAE: {ma_mae}')

# Holt's Linear Trend Model
holt_model = ExponentialSmoothing(data['Target'], trend='add').fit()
data['Holt_Forecast'] = holt_model.fittedvalues

# Calculate MAE for Holt's Linear Trend
holt_mae = np.mean(np.abs(data['Target'] - data['Holt_Forecast']))
print(f'Holt's Linear Trend MAE: {holt_mae}')

# Holt-Winters Model (Additive Model Example)
holt_winters_model = ExponentialSmoothing(data['Target'], trend='add', seasonal='add', seasonal_periods=12).fit()
data['Holt_Winters_Forecast'] = holt_winters_model.fittedvalues

# Calculate MAE for Holt-Winters Model
holt_winters_mae = np.mean(np.abs(data['Target'] - data['Holt_Winters_Forecast']))
print(f'Holt-Winters MAE: {holt_winters_mae}')

# ARIMA Model
arima_order = (4, 1, 0)  # Example order (p, d, q)
arima_model = ARIMA(data['Target'], order=arima_order).fit()
data['ARIMA_Forecast'] = arima_model.fittedvalues

# Calculate MAE for ARIMA Model
arima_mae = np.mean(np.abs(data['Target'] - data['ARIMA_Forecast']))
print(f'ARIMA MAE: {arima_mae}')

# Comparison of Models
print(f'Naive MAE: {naive_mae}')
print(f'Moving Average MAE: {ma_mae}')
print(f'Holt's Linear Trend MAE: {holt_mae}')
print(f'Holt-Winters MAE: {holt_winters_mae}')
print(f'ARIMA MAE: {arima_mae}')
