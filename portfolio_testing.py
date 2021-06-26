# Script to perform back-testing, out-of-sample testing and metrics calculations

import portfolio_optimization
import numpy as np
import pandas as pd

# Function to calculate P&L, we assume that its possible to buy fractions of shares 
def calc_mv(initial, weights, prices):
    '''
    INPUTS
    initial = initial investment 
    weigths = array of weigths, sum should be 1
    prices = DataFrame with prices, same # of instruments and in the same position as in weights
    OUTPUT
    Market_value = DataFrame with market value by asset, and "portfolio" series
    '''
    
    # Get the number of shares of each asset
    shares = weights * initial / prices.loc[0].values
    
    # Calculate market value by instrument, and portfolio
    market_value = prices * shares     
    market_value['portfolio'] = market_value.sum(axis=1)
    
    # Calculate absolute returns
    abs_ret = pd.DataFrame(index=market_value.index)
    abs_ret['Abs_ret'] = market_value['portfolio'] / initial - 1
    
    return shares, market_value, abs_ret


# Calculate historical VaR
def hist_var(shares,prices,alpha):
    '''
    INPUTS
    shares = current units/number of shares/position of each instrument
    prices = historical prices of the instruments
    alpha = level of confidence, number between 0 and 100
    '''
    
    price_change = prices.diff()
    daily_pnl = price_change * shares 
    portfolio_pnl = daily_pnl.sum(axis=1)
    var = portfolio_pnl.quantile(q=(1-alpha/100), interpolation='lower')
    return var


