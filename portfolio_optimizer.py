from scipy.optimize import minimize
import math

import numpy as np
import pandas as pd

# Portfolio Optimization class
class portfolio_optimizer:
    """
    This class is aimed to apply various optimization algorithms to find optimal weight of our Global macro portfolios.

    =====Attributes=====
    semiannual: dataframe of semiannual returns for each ETFs
    """

    def __init__(self, returns):

        self.semiannual = returns

    def portfolio_simulator(self, initial_capital, riskfree, top_cor, cutoff, optimization_type):

        # Risk Parity:
        def Risk_parity(portfolio, kappa=5):
            # min   (1/2) * (x' * Q * x) - kappa * sum_{i=1}^n ln(x_i)
            # s.t.  x >= 0

            _cov = portfolio.cov()
            # number of assets
            n = _cov.shape[0]

            # parameters for nonlinear program
            eps = 1e-3

            def safe_log(x):
                if x > eps:
                    return math.log(x)
                logeps = math.log(eps)
                a = 1 / (3 * eps * (3 * logeps * eps) ** 2)
                b = eps * (1 - 3 * logeps)
                return a * (x - b) ** 3

            def f(x, _cov, kappa):
                func = 0.5 * x.T.dot(_cov.dot(x))

                for i in range(n):
                    func = func - kappa * safe_log(x[i])

                return func

            xinit = np.repeat(1 / n, n)
            lb = 0
            ub = 1
            bnds = tuple([(lb, ub) for x in xinit])

            res = minimize(f, args=(_cov, kappa), method='trust-constr',
                           x0=xinit, bounds=bnds, tol=10 ** -3)

            return res.x / np.sum(res.x)

        # Maximum Sharpe Ratio Portfolio
        def sharpe_ratio_maximization(portfolio):

            def sharpe_ratio_calculation(w, _return, _cov):
                return - _return.mul(w, axis=1).sum(axis=1).cumsum()[-1] / np.sqrt(np.dot(w.T, np.dot(_cov, w)))

            _return = portfolio
            _cov = portfolio.cov()
            n = _cov.shape[0]
            constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
            outcome = minimize(sharpe_ratio_calculation, x0=np.repeat(1 / n, n), args=(_return, _cov),
                               constraints=constraints, bounds=tuple((0, 1) for x in range(n)))
            return outcome.x

        # 50/50 for US and CA ETF
        USDcapital = initial_capital / 2
        CADcapital = initial_capital / 2
        capital = USDcapital + CADcapital

        # dataframes to store the portfolio info
        dollar_full_portfolio = pd.DataFrame()
        pct_full_portfolio = pd.DataFrame()
        USD_PnL = {}
        CAD_PnL = {}
        PnL = {}

        for i in range(2, len(self.semiannual.keys())):

            # Calculate the optimal weights

            # First period correlation is for next period.
            usd = self.semiannual[i - 1][top_cor[i - 1][:cutoff]].subtract(riskfree['RFR'], axis=0)
            cad = self.semiannual[i - 1][top_cor[i - 1][cutoff:]].subtract(riskfree['RFR'], axis=0)

            # fit optimization based on specified type
            if optimization_type == 'Risk parity':
                usdweights = Risk_parity(usd)
                cadweights = Risk_parity(cad)
            elif optimization_type == 'Sharpe ratio maximization':
                usdweights = sharpe_ratio_maximization(usd)
                cadweights = sharpe_ratio_maximization(cad)
            else:
                raise Exception('Type not available.')

            # Calculate the returns
            usdreturns = self.semiannual[i][top_cor[i - 1][:cutoff]]
            cadreturns = self.semiannual[i][top_cor[i - 1][cutoff:]]

            usd_portfolio_return = usdreturns.mul(usdweights, axis=1)
            cad_portfolio_return = cadreturns.mul(cadweights, axis=1)

            # Remove an outlier
            usd_portfolio_return[usd_portfolio_return.values > 100] = 0
            cad_portfolio_return[cad_portfolio_return.values > 100] = 0

            # Portfolio Returns
            portfolio_return = pd.concat([usd_portfolio_return, cad_portfolio_return], axis=1, ignore_index=True)

            pct_full_portfolio = pd.concat([pct_full_portfolio, portfolio_return])
            dollar_full_portfolio = pd.concat([dollar_full_portfolio, portfolio_return])

            # PnL calculation
            USD_PnL[i] = usd_portfolio_return.sum(axis=1) * USDcapital
            CAD_PnL[i] = cad_portfolio_return.sum(axis=1) * CADcapital
            PnL[i] = USD_PnL[i] + CAD_PnL[i]
            capital += (PnL[i].cumsum()[-1] - PnL[i].cumsum()[0])
            USDcapital = capital / 2
            CADcapital = capital / 2

            # Note: we don't need to do rebalancing operation. Just reallocate the capital is enough.

        return pct_full_portfolio, dollar_full_portfolio, PnL
