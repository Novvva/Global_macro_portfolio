import numpy as np
import math
import os
import FX_Volatility
from scipy.stats.distributions import chi2
import pandas as pd
import matplotlib.pyplot as plt
import random


# portfolio risk analytics class

class risk:
    """
    This class aimed to generate various risk analytic metrics for our portfolios.
    ___Attributes___
    : Dataframe that contains each individual factors return
    : weights from portfolio_optimization
    : # of factors
    """

    def __init__(self, factReturns, weight, num):
        self.weight = weight
        self.mean = factReturns.mean()
        self.cov = factReturns.cov()
        self.variance = np.sqrt(np.dot(weight.T, np.dot(factReturns.cov(), weight)))
        self.num = num
        self.factReturns = factReturns

    # MC_multivariate normal structure VaR&CVaR calculation
    def VaR_CVaR_MC(self, numTrials, alpha=5):

        # We assume the underlying ETF distribution follow a multivariate-normal structure with
        # corresponding correlation matrix.
        factorCov = self.factReturns.cov()
        factormean = self.mean

        trialReturns = []
        for i in range(0, numTrials):

            # Generate the sample of factor returns
            trialFactorReturns = np.random.multivariate_normal(factormean, factorCov)

            # Calculate the returns of each instrument, and then calculate teh total returns for this trial.
            trailTotalReturn = np.dot(self.weight, trialFactorReturns)

            trialReturns.append(trailTotalReturn)

        _return = np.array(trialReturns)
        VaR = np.percentile(_return, alpha)
        cVaR = _return[_return <= VaR].mean()
        return VaR, cVaR

    # Plot the VaR & CVaR using historical & method method.
    def plot_VaR(self, trail):
        _alpha = np.linspace(0,100,100)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
        mc_VaR_list = []
        mc_cVaR_list = []
        hist_VaR_list = []
        hist_cVaR_list = []

        for alpha in _alpha:
            _varmc, _cvarmc = self.VaR_CVaR_MC(trail, alpha=alpha)
            _varhist, _cvarhist = self.hist_var(alpha=alpha)
            mc_VaR_list.append(_varmc)
            mc_cVaR_list.append(_cvarmc)
            hist_VaR_list.append(_varhist)
            hist_cVaR_list.append(_cvarhist)

        ax[0].plot(_alpha, mc_VaR_list, label="MC-method")
        ax[0].plot(_alpha, hist_VaR_list, label="historical-method")
        ax[0].set_xlabel('Alpha')
        ax[0].set_ylabel('VaR')
        ax[0].set_title('VaR Estimation')
        ax[0].legend()
        ax[1].plot(_alpha, hist_cVaR_list, label="MC-method")
        ax[1].plot(_alpha, mc_cVaR_list, label='historical-method')
        ax[1].set_xlabel('Alpha')
        ax[1].set_ylabel('cVaR')
        ax[1].set_title('cVaR Estimation')
        ax[1].legend()

    # MC-simulation by bootstrapping historical data.
    def hist_var(self, alpha=5):
        _portfolio_return = self.factReturns.mul(self.weight,axis=1).apply(lambda x: x.sum(), axis=1)
        VaR = _portfolio_return.quantile(q=alpha/100, interpolation='lower')
        CVaR = _portfolio_return[_portfolio_return <= VaR].mean()
        return VaR, CVaR

    # Calculate the maximum_drawdown of the final portfolio
    def max_drawdown(self):
        _portfolio_return = self.factReturns.mul(self.weight,axis=1).apply(lambda x: x.sum(), axis=1)

        # Generate the cumulative gross return index
        wealth_index = (1+_portfolio_return).cumprod()*1000
        previous_peaks = wealth_index.cummax()

        # Drawdown = current_value / Previous peaks
        drawdown = (wealth_index - previous_peaks)/previous_peaks
        return drawdown.min()

    # Calculate the Sharpe ratio of the final portfolio
    def sharpe_ratio(self):
        _portfolio_return = self.factReturns.mul(self.weight, axis=1).sum(axis=1).cumsum()[-1]
        _portfolio_vol = self.factReturns.mul(self.weight, axis=1).apply(lambda x: x.sum(), axis=1).std()
        _sharpe_ratio = _portfolio_return/_portfolio_vol
        _real_ratio = _sharpe_ratio*np.sqrt(2)
        return _real_ratio


class scenario_analysis:
    """
       This class aimed to generate various scenario testings for our portfolios.
       ___Attributes___

       """
    def __init__(self, returns):
        self.returns = returns

    # The market has 2% probability to crash together in a day.
    def all_crash(self):
        all_crash_returns = self.returns.copy()
        for i in range(len(self.returns.index)):
            rand = random.random()
            if rand < 0.02:

                # have 2% probability that the market will crash by 50%
                all_crash_returns.iloc[i] = self.returns.iloc[i]/2 

        return all_crash_returns

    # The portfolio will randomly crash for some reason
    def random_crash(self):
        random_crash_returns = self.returns.copy()
        for i in range(len(self.returns.index)):
            rand_prob = np.random.rand(len(random_crash_returns.columns))

            # have 5% probability that each ETF will crash by a half
            for j in range(len(rand_prob)):
                if rand_prob[j] < 0.05 and self.returns.iloc[i, j] > 0:
                    random_crash_returns.iloc[i, j] = self.returns.iloc[i, j] / 2
                elif rand_prob[j] < 0.05 and self.returns.iloc[i, j] < 0:
                    random_crash_returns.iloc[i, j] = self.returns.iloc[i, j] * 2

        return random_crash_returns

    #Given that yesterday crash, the market has a moving-average effect to crash together.
    def mv_crash(self, lag, correlation):
        def crash_together(series):
            if series['default'] == True:
                series /= 2
            return series

        mv_crash_returns = self.returns.copy()
        mv_crash_returns['default'] = False
        for i in range(len(self.returns.index)):
            rand = random.random()
            _value = rand
            total = 1

            for j in range(max(i-lag,0),i):
                _value += correlation **(i-j) * mv_crash_returns.iloc[j,-1]
                total += correlation ** (i-j)
    
             # Normalization
            default = _value/total
            if default > 0.75:
                mv_crash_returns.iloc[i,-1] = True
            else:
                mv_crash_returns.iloc[i,-1] = False
        mv_crash_returns = mv_crash_returns.apply(crash_together,axis=1)
        return mv_crash_returns.drop(['default'], axis=1)
        
# Helper function
def create_new_profile(old_profile,_return):
    new_profile = old_profile.copy()
    new_profile.factReturns = _return
    return new_profile

class option_risk:
    pass