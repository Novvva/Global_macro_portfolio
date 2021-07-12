import numpy as np
import math
import os
#import portfolio_optimization
import FX_Volatility
#from qpsolvers import solve_qp
from scipy.stats.distributions import chi2
import pandas as pd
import matplotlib.pyplot as plt


# portfolio risk analytics class

class risk:
    """
    This class aimed to generate various risk analytic metrics for our portfolios.
    ___Attributes___
    : Dataframe that contains each individual factors return
    : weights from portfolio_optimization
    : # of factors
    """

    def __init__(self,factReturns,weight,num):
        self.weight = weight
        self.mean = factReturns.mean()
        self.cov = factReturns.cov()
        self.variance = np.sqrt(np.dot(weight.T, np.dot(factReturns.cov, weight)))
        self.num = num
        self.returns = factReturns

    # MC_multivariate normal structure VaR&CVaR calculation
    def VaR_CVaR_MC(self,numTrials,alpha=5):

        # We assume the underlying ETF distribution follow a multivariate-normal structure with corresponding correlation matrix.
        _cov = self.returns.cov()
        _mean= self.mean

        trialReturns = []
        for i in range(0,numTrials):

            # Genearte the sample of factor returns
            trialFactorReturns = np.random.multivariate_normal(factormean,factorCov)

            #Calcualte the returns of each instrument, and then calculate teh total returns for this trial.
            trailTotalReturn = np.dot(self.weight,trialFactorReturns)

            trialReturns.append(trailTotalReturn)

        _return = np.array(trialReturns)
        VaR = np.percentile(_return,alpha)
        cVaR = _return[_return<=VaR].mean()
        return VaR,cVaR

    # Plot the VaR & CVaR using histocial & method method.
    def plot_VaR(self,trail):
        _alpha = np.linspace(0,100,100)
        fig,ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10,10))
        mc_VaR_list = []
        mc_cVaR_list = []
        hist_VaR_list = []
        hist_cVaR_list = []

        for alpha in _alpha:
            _varmc,_cvarmc = self.VaR_CVaR_MC(trail,alpha = alpha)
            _varhist,_cvarhist = self.hist_var(alpha = alpha)
            mc_VaR_list.append(_varmc)
            mc_cVaR_list.append(_cvarmc)
            hist_VaR_list.append(_varhist)
            hist_cVaR_list.append(_cvarhist)

        ax[0].plot(_alpha,mc_VaR_list,label = "MC-method")
        ax[0].plot(_alpha,hist_VaR_list,label = "historical-method")
        ax[0].set_xlabel('Alpha')
        ax[0].set_ylabel('VaR')
        ax[0].set_title('VaR Estimation')
        ax[0].legend()
        ax[1].plot(_alpha,hist_cVaR_list,label = "MC-method")
        ax[1].plot(_alpha,mc_cVaR_list,label = 'historical-method')
        ax[1].set_xlabel('Alpha')
        ax[1].set_ylabel('cVaR')
        ax[1].set_title('cVaR Estimation')
        ax[1].legend()

    # MC-simulation by bootstrapping historical data.
    def hist_var(self,alpha=5):
        _portfolio_return = self.factReturns.mul(self.weight,axis=1).apply(lambda x: x.sum(),axis=1)
        VaR = _portfolio_return.quantile(q = alpha/100,interpolation = 'lower')
        CVaR = _portfolio_return[_portfolio_return<=VaR].mean()
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

    def sharpe_ratio(self,rf):
        _portfolio_return = self.factReturns.mul(self.weight,axis=1).apply(lambda x: x.sum(),axis=1) - rf
        wealth_index = (1+_portfolio_return).cumprod()
        _portfolio_vol = self.factReturns.mul(self.weight,axis=1).apply(lambda x: x.sum(),axis=1).std()
        return wealth_index[-1]/_portfolio_vol


class scenario_analysis:
    """
       This class aimed to generate various scneario testing for our portfolios.
       ___Attributes___
       : previous risk_profile
       """
    def __init__(self,risk_profile):
        self.profile = risk_profile

    # The market has 2% probabily to crash together in a day.
    def all_crash(self):
        def crash_together(series):
            rand = random.random()
            if rand <0.02:
                series -= 0.005
            return series
        R = self.risk_profile.factReturns.apply(lambda x: random_crash(x),axis=1)
        return create_new_profile(self.profile,R)

    # The portfolio will randomly crash for some reason
    def Random_crash(self):
        def random_crash(x):
            rand = random.random()
            if rand < 0.05 and x>0:
                x /=2
            elif rand < 0.05 and x <0:
                x *= 2
            return x
        R = self.risk_profile.factReturns.applymap(lambda x: random_crash(x))
        return create_new_profile(self.profile,R)

    # Given that yesterday crash, the market has a moving-average effect to crash together.
    def mv_crash(self,lag,correlation):
        def crash_together(series):
            if series['default'] == True:
                series -= 0.005
            return series

        _newprofile = self.profile.copy()
        _newprofile.factReturns['default'] = False
        
        for i in range(x.shape(0)):
            rand = random.random()
            _value = rand
            total = 1

            # Given the correlation and the lagging coefficient, the default risk is moving average.
            for j in range(max(i-lag,0),i):
                _value += correlation **(i-j) * _newprofile[j,-1]
                total += correlation ** (i-j)

            # Normalization
            default = _value/total
            if default > 0.95:
                _newprofile[i,-1] = True
            else:
                _newprofile[i,-1] = False

        _newprofile.factReturns = _newprofile.factReturns.apply(lambda x: crash_together(_newprofile.factReturns),axis=1)
        return _newprofile

# Helper function
def create_new_profile(old_profile,_return):
    new_profile = old_profile.copy()
    new_profile.factReturns = _return
    return new_profile

class option_risk:
    pass