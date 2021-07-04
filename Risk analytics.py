import numpy as np
import math
import os
import portfolio_optimization

from qpsolvers import solve_qp
from scipy.stats.distributions import chi2

# portfolio risk analytics class

class Risk_analytics:
    """
    This object aimed to generate various risk analytic metrics for our portfolios.
    ___Attributes___
    : Dataframe that contains portfolio series return and each individual return
    : factorReturns
    : weights from portfolio_optimization
    : # of factors
    """

    def __init__(self,returns,factReturns,weight,num):
        self.returns = returns
        self.weight = weight
        self.mean = returns.mean()
        self.variance = np.sqrt(np.dot(weight.T,np.dot(self.returns.cov(),wight)))
        self.num = 58
        self.factReturns = factReturns

    # MC_multivariate normal structure VaR&CVaR calculation
    def VaR_CVaR_MC(self,numTrials,alpha=5):

        # We assume the underlying distribution follow a multivariate-normal structure with corresponding correlation matrix.
        factorCov = np.cov(self.factReturns)
        factormean = sum(self.factReturns[i].mean() for i in range(self.num))

        trialReturns = []
        for i in range(0,numTrials):

            # Genearte the sample of factor returns
            trialFactorReturns = np.random.multivariate_normal(factormean,factorcov)

            #Calcualte the returns of each instrument, and then calculate teh total returns for this trial.
            trailTotalReturn = sum(np.dot(self.weight,trialFactorReturns))

            trialReturns.append(trailTotalReturn)

        _return = np.array(trialReturns)
        VaR = np.percentile(_return,alpha)
        cVaR = _return[_return<VaR].mean()
        return VaR,cVaR

    # MC-simulation by bootstrapping historical data.
    def hist_var(self,alpha=5):
        _portfolio_return = self.factReturns.mul(self.weight,axis=1)
        VaR = _portfolio_return.quantile(q = (1-alpha/100),interpolation = 'lower')
        CVaR = _portfolio_return[_portfolio_return<VaR].mean()
        return VaR, CVaR

    def max_drawdawn(self):
        _portfolio_return = self.factReturns.mul(self.weight,axis=1)

        # Generaet teh cumulative gross return index
        wealth_index = (1+_portfolio_return).cumprod()*1000
        previous_peaks = wealth_index.cummax()

        # Drawdown = current_value / Previous peaks
        drawdown = (wealth_index - previous_peaks)/previous_peaks
        return drawdown.min()

class risk_reporting:
    def __init__(self):
        pass

class scenario_analysis:
    def __init__(self):
        pass

    def bear_market(self):
        pass

    def market_crash(self):
        pass

    def correlation_switch(self):
        pass










