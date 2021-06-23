import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Portfolio Optimization class

class portfolio_optimization:
    """
    This class is aimed to apply various optimization algorithms to find optimal weight of our Global macro portfolios.

    =====Attributes=====
    returns: asset returns

    factReturns: factor returns

    Q: asset covariance matrix

    mu: asset predicted return

    """

    def __init__(self, returns, factReturns, model_type, lam=0.01, K=7):
        """
        Initialize the portfolio optimization.
        :param returns: dataframe that contains the returns of all our assets.
        :param factReturns: dataframe that contains all factorial returns.
        :param model_type: regression model type to find Q and mu.
        :param lam: lambda parameter for LASSO, default = 0.01.
        :param K: K parameter for BSS, default = 7.
        """
        self.returns = returns
        self.factReturns = factReturns

        if model_type == 'OLS':
            # implement Ordinary Least Square
            x = factReturns.copy()

            # Add the constant line
            X = x.insert(0,'constant',np.ones(1))
            # Optimize the reisudals
            B = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(returns)
            # Calculate the residuals
            e = returns.values - X.T.dot(B)
            # Calculate the variance of residuals
            ve = e.T.dot(e)/(X.shape[0]-X.shape[1]-1)
            # The vector of expected asset return
            f = factReturns.mean()
            _f = f.copy().insert(0, 'constant', np.ones(1))
            # Calculate the Factor Covariance matrix
            F = factReturns.cov()

            self.Q =  np.delete(B,0,axis=1).T.dot(F).dot(np.delete(B,0,axis=1)) + np.diag(np.diag(ve))
            self.mu = B.T.dot(_f)

        elif model_type == 'FF':
            # implement Fama French

            self.Q =
            self.mu =

        elif model_type == 'LASSO':
            # implement LASSO regression

            self.Q =
            self.mu =

        elif model_type == 'BSS':
            # implement Best subset selections

            self.Q =
            self.mu =

        else:
            raise Exception('model type is not defined!')

    def MVO(self):
        """
        Perform Mean Variance Optimization.
        :return: weights of assets
        """

        return x

    def robust_MVO(self, alpha, lam):
        """
        Perform robust MVO.
        :param alpha:
        :param lam:
        :return: weights of assets
        """

        return x

    def RP(self, kappa):
        """
        Perform Risk Parity.
        :param kappa:
        :return: weights of assets
        """

        return x

    def CVaR_optimization(self, alpha):
        """
        Perform CVaR optimization
        :param alpha:
        :return: weights of assets
        """

        return x