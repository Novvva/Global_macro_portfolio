# File List:

* CAN_ETFs.csv - List of Canadian ETFs used for our algorithm
* USA_ETFs.csv - List of American ETFs used for our algorithm 
* Currency_Spot.xlsx - List of Currencies and their spot prices used
* Portfolio_Construction.ipynb - Using the input data to construct our portfolios, contains some risk mgt
* data_preprocessing.py - Scrapes the data using the tickers from the ETF .csv files from Yahoo finance, calculates the % returns, and returns a .csv of each asset’s return stream
* returns.csv/.xlsx - Output from data_preprocessing
* vol_data.csv - Volatility prices database of 8 currencies, EURUSD, USDCAD, USDMXN, USDBRL, USDZAR, USDRUB, USDCNY, USDJPY. The quotes used are spot price, forward points, At-the-money vol, risk reversal and butterflies with maturities 1w, 1m and 3m. With these inputs we can calibrate a vol surface for each currency. 
* FX_Volatility.py - This script has all the functions necessary to calculate the P&L of an options strategy. It works creating 3 objects, the option (prices and sensitivities), the volatility surface and the P&L of any options strategy. We are considering 4 possible strategies:
	1.	Put spread delta 30-20
	2.	Put spread delta 40-30
	3.	Call Spread 40-30 
	4.	Call Spread 30-20
The code is calibrating a volatility surface for each day, using a parabolic interpolation on the volatility smile. First, we calculate the vol surface using a moneyness approach and then we change it to strike-tenor and calibrate it. We are using a bilinear interpolation in the final surface to get the volatility of a given strike or a given delta. 

* Portfolio Optimization.py
	* Using factor models (Linear and LASSO) to predict expected return and covariance matrix 
	* Using optimization models (Minimum Variance Optimization (MVO), robust MVO, Risk Parity, CVaR optimization) to predict the optimal weights.

## PnL calculation function:


* portfolio_testing.py
	*Function:
    calc_mv(initial, weights, price):  Function to calculate P&L, assuming that its possible to buy fractions of shares. 

## Risk management and analytics function:


* Risk_analytics.py

	* Class risk:  the class aimed to generate various risk analytic metrics for the portfolios. 
	* VaR_CVaR_MC(self,numTrials,alpha):   Calculate the VaR&CVaR through standard Monte Carlo simulation, assuming the underlying ETF list follows a multi-normal distribution. 
	* hist_var(self,alpha): Calculate the VaR&CVaR through historical approach. 
	* plot_VaR(self,trail):  Plot the VaR & CVaR curve by both two methods, under different quantiles. 
	* max_drawdown(self):  Calculate the maximum_drawdown ratio of a given period. 
	* sharpe_ratio(self,rf): Calculate the sharpe-ratio of a given period. 
	* Class scneraio_anlysis:  the class aimed to generate stress testing scenarios for the portfolios. 
	* all_crash(self): the scenario that the market has 2% probability to crash together in a day. 
	* Random_crash(self):  the scenario that each ETF in the portfolio has 5% probability to crash in a day. 
	* mv_crash(self,lag,correlation): the scenario that given the info of the lagging period, the market has more probability to crash.
  
 
  



