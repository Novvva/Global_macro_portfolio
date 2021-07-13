from scipy.optimize import minimize
import math

import numpy as np
import pandas as pd

import Risk_analytics

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

        # MVO
        def MVO(portfolio, targetRet=None):
            """
            Perform Mean Variance Optimization.
            :return: weights of assets
            """
            _mu = portfolio.mean(axis=0).to_numpy()
            _cov = portfolio.cov().to_numpy()

            # total number of assets
            n = _cov.shape[0]
            if targetRet is None:
                targetRet = np.mean(_mu)

            def f(x, Q):
                func = np.matmul(np.matmul(x, Q), x.T)
                return func

            def constraintEq(x):
                Aeq = np.ones(x.shape)
                beq = 1
                EqconstraintVal = np.matmul(Aeq, x.T) - beq
                return EqconstraintVal

            def constraintIneq(x, mu, targetRet):
                Aneq = np.array(mu)
                bneq = targetRet
                IneqconstraintVal = np.matmul(Aneq, x.T) - bneq
                return IneqconstraintVal

            xinit = np.repeat(1 / n, n)
            cons = ({'type': 'eq', 'fun': constraintEq},
                    {'type': 'ineq', 'fun': constraintIneq, 'args': (_mu, targetRet)})
            lb = 0
            ub = 1
            bnds = tuple([(lb, ub) for x in xinit])

            res = minimize(f, args=(_cov), method='trust-constr',
                           x0=xinit, bounds=bnds, constraints=cons, tol=10 ** -3)

            return res.x

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

        USD_risk_profile = pd.DataFrame()
        CAD_risk_profile = pd.DataFrame()
        overall_risk_profile = pd.DataFrame()

        for i in range(2, len(self.semiannual.keys())):

            # Calculate the optimal weights

            # First period correltion is for next period.
            usd = self.semiannual[i - 1][top_cor[i][:cutoff]].subtract(riskfree['RFR'], axis=0)[
                  self.semiannual[i - 1][top_cor[i][:cutoff]].index[0]:self.semiannual[i - 1][top_cor[i][:cutoff]].index[-1]]
            cad = self.semiannual[i - 1][top_cor[i][cutoff:]].subtract(riskfree['RFR'], axis=0)[
                  self.semiannual[i - 1][top_cor[i][cutoff:]].index[0]:self.semiannual[i - 1][top_cor[i][cutoff:]].index[-1]].fillna(0)

            # fit optimization based on specified type
            if optimization_type == 'MVO':
                usdweights = MVO(usd)
                cadweights = MVO(cad)
            elif optimization_type == 'Risk parity':
                usdweights = Risk_parity(usd)
                cadweights = Risk_parity(cad)
            elif optimization_type == 'Sharpe ratio maximization':
                usdweights = sharpe_ratio_maximization(usd)
                cadweights = sharpe_ratio_maximization(cad)
            else:
                raise Exception('Type not available.')

            # Calculate the returns
            usdreturns = self.semiannual[i][top_cor[i][:cutoff]]
            cadreturns = self.semiannual[i][top_cor[i][cutoff:]]

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

            # Risk calculation
            _USDrisk = Risk_analytics.risk(usdreturns, usdweights, cutoff)
            _CADrisk = Risk_analytics.risk(cadreturns, cadweights, cutoff)
            _USDVaR95, _USDCVaR95 = _USDrisk.hist_var(5)
            _CADVaR95, _CADCVaR95 = _CADrisk.hist_var(5)
            _USDVaR99, _USDCVaR99 = _USDrisk.hist_var(1)
            _CADVaR99, _CADCVaR99 = _CADrisk.hist_var(1)
            _USDVaR95 *= USDcapital
            _USDVaR99 *= USDcapital
            _USDCVaR95 *= USDcapital
            _USDCVaR99 *= USDcapital
            _CADVaR95 *= CADcapital
            _CADVaR99 *= CADcapital
            _CADCVaR95 *= CADcapital
            _CADCVaR99 *= CADcapital

            _USDriskprofile = {'Period': i, 'VaR 95%': [_USDVaR95], 'VaR 99%': [_USDVaR99],
                               'CVaR 95%': [_USDCVaR95], 'CVaR 99%': [_USDCVaR99],
                               'Sharpe Ratio': [_USDrisk.sharpe_ratio()], 'Max_drawdown': [_USDrisk.max_drawdown()]}

            _CADriskprofile = {'Period': i, 'VaR 95%': [_CADVaR95], 'VaR 99%': [_CADVaR99],
                               'CVaR 95%': [_CADCVaR95], 'CVaR 99%': [_CADCVaR99],
                               'Sharpe Ratio': [_CADrisk.sharpe_ratio()], 'Max_drawdown': [_CADrisk.max_drawdown()]}

            _overallriskprofile = {'Period': i, 'VaR 95%': [_USDVaR95 + _CADVaR95],
                                   'VaR 99%': [_USDVaR99 + _CADVaR99],
                                   'CVaR 95%': [_USDCVaR95 + _CADCVaR95],
                                   'CVaR 99%': [_USDCVaR99 + _CADCVaR99]}

            _USDriskprofile = pd.DataFrame.from_dict(_USDriskprofile)
            _CADriskprofile = pd.DataFrame.from_dict(_CADriskprofile)
            _overall_risk_profile = pd.DataFrame.from_dict(_overallriskprofile)
            USD_risk_profile = pd.concat([USD_risk_profile, _USDriskprofile])
            CAD_risk_profile = pd.concat([CAD_risk_profile, _CADriskprofile])
            overall_risk_profile = pd.concat([overall_risk_profile, _overall_risk_profile])

            # Note: we don't need to do rebalancing operation. Just reallocate the capital is enough.

        USD_risk_profile.set_index('Period', inplace=True)
        CAD_risk_profile.set_index('Period', inplace=True)
        overall_risk_profile.set_index('Period', inplace=True)

        # max drawdown:
        ts = pd.concat(PnL).cumsum()
        previous_peaks = ts.cummax()
        drawdown = (ts - previous_peaks) / previous_peaks

        max_drawdown = f'{drawdown.min()*100}%'

        return pct_full_portfolio, dollar_full_portfolio, PnL, USD_risk_profile, CAD_risk_profile, overall_risk_profile, max_drawdown
