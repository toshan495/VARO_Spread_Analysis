import os

import matplotlib.pyplot as plt
import pandas as pd
import ruptures as rpt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

current_directory = os.getcwd()
##print("Current working directory:", current_directory)

brent_symbol = 'BZ=F'  # Brent crude oil futures
wti_symbol = 'CL=F'  # WTI crude oil futures
start_date = '2010-01-01'
end_date = '2023-08-01'
brent_data = yf.download(brent_symbol, start=start_date, end=end_date)['Adj Close']
wti_data = yf.download(wti_symbol, start=start_date, end=end_date)['Adj Close']
spread = brent_data - wti_data
ratio = brent_data / wti_data

# Create a single DataFrame for both Brent and WTI
combined_data = pd.DataFrame({
    'Brent_Adj_Close': brent_data,
    'WTI_Adj_Close': wti_data,
    'Spread': spread,
    'Ratio': ratio
})

# excel_filename = 'brent_wti_data.xlsx'
# combined_data.to_excel(excel_filename, index=True)
# Print the combined data
# print(combined_data)
# print(spread.mean())

plt.figure(figsize=(10, 6))
plt.plot(spread.index, spread, color='blue', label='Brent-WTI Spread')
plt.plot(brent_data.index, brent_data, color='red', label='Brent')
plt.plot(wti_data.index, wti_data, color='green', label='WTI')
plt.plot(ratio.index, ratio, color='yellow', label='Ratio')
plt.title('Brent-WTI Spread from 2010 to 2023')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

adf_result = adfuller(spread.dropna(), autolag='AIC')

# Print the ADF test results
print("ADF Test Results:")
print(f"ADF Statistic: {adf_result[0]}")
print(f"P-value: {adf_result[1]}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value}")

# Plot the Brent-WTI spread
plt.figure(figsize=(10, 6))
plt.plot(spread.index, spread, label='Brent-WTI Spread', color='blue')
plt.axhline(y=0, color='red', linestyle='--', label='Zero Spread')
plt.title('Brent-WTI Spread and ADF Test (2010-2023)')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.grid(True)
plt.legend()


# plt.show()


################ Change Point Detection

def detect_changes(time_series):
    # Convert time series to a numpy array
    signal = time_series.values
    # Perform change point detection using the Pelt algorithm
    algo = rpt.Pelt(model="rbf", min_size=1, jump=10).fit(signal)
    result = algo.predict(pen=2)
    # remove location if equal to len(signal)
    change_points = [i for i in result if i < len(signal)]
    # Return the list of change point locations
    return change_points


signal = spread
changes = detect_changes(signal)
# Plot the time series
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(signal, label='signal')
# Plot the abrupt changes
for i in changes:
    ax.axvline(signal.index[i], color='r')

# Add labels, grid, and legend
ax.set_xlabel('Time')
ax.set_ylabel('Signal')
ax.set_title('Time Series with Abrupt Changes')
ax.legend()
ax.grid(True)

# Show the plot
plt.show()
