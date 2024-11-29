#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:43:51 2024

@author: ishratvasid
"""

import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Fetch Apple's historical stock data
start_date = "2014-10-31"
end_date = "2024-10-31"
aapl_data = yf.download("AAPL", start=start_date, end=end_date)
aapl_data['Returns'] = aapl_data['Adj Close'].pct_change()
aapl_data =aapl_data.dropna()

# Step 2: Fetch Fama-French 3-factor data
# Download Fama-French data from Kenneth French's library or preloaded CSV
fama_french_data = pd.read_csv("/Users/ishratvasid/Desktop/DESIM/AAPL/F-F_Research_Data_Factors_daily.CSV", skiprows=3, skipfooter=1, engine='python')
fama_french_data = fama_french_data.rename(columns={"Unnamed: 0": "Date"})
fama_french_data['Date'] = pd.to_datetime(fama_french_data['Date'], format='%Y%m%d')
fama_french_data.set_index('Date', inplace=True)

# Convert percentages to decimal
for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
    fama_french_data[col] = fama_french_data[col] / 100

# Step 3: Merge the data
aapl_data = aapl_data.reset_index()
aapl_data['Date'] = pd.to_datetime(aapl_data['Date'])
merged_data = pd.merge(aapl_data, fama_french_data, on='Date', how='inner')

# Calculate excess returns
merged_data['Excess_Returns'] = merged_data['Returns'] - merged_data['RF']

# Step 4: Perform Regression
X = merged_data[['Mkt-RF', 'SMB', 'HML']]
y = merged_data['Excess_Returns']
X = sm.add_constant(X)  # Add a constant for the intercept

model = sm.OLS(y, X).fit()
print(model.summary())

# Step 5: Plot results
plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, y, alpha=0.5)
plt.plot(y, y, color='red', label='45-degree line')
plt.xlabel("Fitted Values (Predicted Excess Returns)")
plt.ylabel("Actual Excess Returns")
plt.title("Fitted vs Actual Excess Returns")
plt.legend()
plt.show()