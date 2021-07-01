# Script to perform data cleaning and preprocessing

import yfinance as yf
import numpy as np
import pandas as pd
import os
import sys

__file__ = 'data_preprocessing.py'

# Set working directory to file save location
os.getcwd()

# Change directory
os.chdir('/Users/lucaskrenn/Documents/MMF2025H-Risk Management Laboratory/Global_macro_portfolio')

# Read in the csv of ETFs
etf = pd.read_csv('list_etf.csv')
list_etf = list(etf['TICKER'].values)

# Add currencies to the list
currs_list = ["CNY=X", "BRL=X", "ZAR=X", "RUB=X", "EUR=X", "JPY=X", "MXN=X"]
list_etf += currs_list


tick_names = []

# Create a dictionary with all the necessary information
prices_dict = {} 
for i in range(len(list_etf)):
    etf = yf.Ticker(list_etf[i])
    prices_dict[list_etf[i]] = etf.history(period="max")
    
    #if list_etf[i] not in currs_list:
    #    tick_names.append([list_etf[i], etf.info['longName']])


# Format nicely and add column names
prices = pd.DataFrame([prices_dict[x]['Close'] for x in prices_dict.keys()]).T
prices.columns = [x for x in prices_dict.keys()]


# Want all the data after Jan 2011
prices = prices[prices.index >= '2011-01-01']

# Calculate percentage change aka daily price returns
rets = prices.pct_change()

# Save to .csv
rets.to_csv('returns.csv')

