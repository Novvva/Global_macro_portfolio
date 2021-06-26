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
#os.chdir('/Users/lucaskrenn/Documents/MMF2025H-Risk Management Laboratory/Global_macro_portfolio')

# Read in the csv of ETFs
etf = pd.read_csv('list_etf.csv')
list_etf = etf['TICKER'].values


# Create a dictionary with all the necessary information
prices_dict = {} 
for i in range(len(list_etf)):
    prices_dict[list_etf[i]] = yf.Ticker(list_etf[i]).history(period="max")
    


# Format nicely and add column names
prices = pd.DataFrame([prices_dict[x]['Close'] for x in prices_dict.keys()]).T
prices.columns = [x for x in prices_dict.keys()]


# Want all the data after Jan 2011
prices = prices[prices.index >= '2011-01-01']

# Calculate percentage change aka daily price returns
rets = prices.pct_change()

# Save to .csv
rets.to_csv('returns.csv')

