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

            self.Q =
            self.mu =

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