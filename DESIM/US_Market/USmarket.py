#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:44:01 2024

@author: ishratvasid
"""

import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Data Collection
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

# Equity indices (S&P 500 and sectors)
equity_tickers = {'S&P 500': '^GSPC',
                  'Technology': 'XLK',
                  'Financials': 'XLF',
                  'Healthcare': 'XLV'}

# Treasury bond proxy (10-year Treasury yield ETF)
bond_ticker = '^TNX'

start_date = '2014-10-31'
end_date = '2024-10-31'

equity_data = {sector: fetch_data(ticker, start_date, end_date) for sector, ticker in equity_tickers.items()}
equity_df = pd.DataFrame(equity_data)
bond_data = fetch_data(bond_ticker, start_date, end_date)
bond_df = pd.DataFrame({'10Y Yield': bond_data})

combined_df = pd.concat([equity_df, bond_df], axis=1).dropna()
returns_df = combined_df.pct_change().dropna() # Calculate returns

# Rolling Correlation
rolling_corr = returns_df.rolling(window=252).corr().unstack()['10Y Yield'].dropna()

# Visualization
plt.figure(figsize=(12, 6))
for sector in equity_tickers:
    plt.plot(rolling_corr.index, rolling_corr[sector], label=f'{sector} vs 10Y Yield')
plt.title('Rolling Correlation Between Equity Sectors and 10Y Yield')
plt.legend()
plt.show()

# Regression Analysis
X = returns_df['10Y Yield']
for sector in equity_tickers:
    Y = returns_df[sector]
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit()
    print(f'Regression Results for {sector}')
    print(model.summary())



