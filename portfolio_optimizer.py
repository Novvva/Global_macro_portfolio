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

    semiannual: dictionary of semiannual returns dataframes for each ETFs in each semi annual periods.

    """

    def __init__(self, returns):

        self.semiannual = returns

    def portfolio_simulator(self, initial_capital, riskfree, top_cor, cutoff, VaRcutoff, optimization_type,
                            benchmark=None, scenario=None):
        """
        :param initial_capital: the assumed initial capital amount for our portfolio.
        :param riskfree: a dataframe of 3-m US treasury yield in each time.
        :param top_cor: top cutoff number of ETFs with lowest correlations.
        :param cutoff: the number of ETFs we want to include in the portfolio.
        :param VaRcutoff: the VaR threshold that is used to determine the scaling factor in each small period.
        :param optimization_type: four different types available: 'MVO', 'Risk parity', 'Sharpe ratio maximization',
                                                                  'Equally weighted'.
        :param benchmark: the benchmark excess returns used for MVO (default=None).
        :return: dollar_full_portfolio: dollar amount for each ETFs in entire portfolio.
                 PnL: Profit and loss.
                 USD_risk_profile: risk metrics for US ETFs.
                 CAD_risk_profile: risk metrics for CA ETFs.
                 overall_risk_profile: risk metrics for entire portfolio.
                 max_drawdown: max drawdown calculation.
        """
        '===================================Optimization Algorithms===================================================='

        '============Mean Variance Optimization============'
        def MVO(portfolio, benchmark=None):
            """
            Perform Mean Variance Optimization.
            :return: weights of assets
            """
            _mu = portfolio.mean(axis=0).to_numpy()
            _cov = portfolio.cov().to_numpy()

            # total number of assets
            n = _cov.shape[0]
            if benchmark is None:
                targetRet = np.mean(_mu)
            else:
                targetRet = benchmark.mean()

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

        '============Risk parity============'
        def Risk_parity(portfolio, kappa=5):
            """
            Perform Risk parity optimization. (Equal Risk Attribution)
            :return: Weights of assets.
            """
            # min   (1/2) * (x' * Q * x) - kappa * sum_{i=1}^n ln(x_i)
            # s.t.  x >= 0

            _cov = portfolio.cov()
            # number of assets
            n = _cov.shape[0]

            # parameters for nonlinear program
            eps = 1e-3

            # avoid log(0)
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

        '============Maximum Sharpe Ratio============'
        def sharpe_ratio_maximization(portfolio):

            _return = portfolio
            _cov = portfolio.cov().to_numpy()
            # number of assets
            n = _cov.shape[0]

            # define maximization of Sharpe Ratio using principle of duality
            def f(x, _return, _cov):
                mu_p = _return.mul(x, axis=1).sum(axis=1).cumsum()[-1]
                cov_p = np.sqrt(np.matmul(np.matmul(x, _cov), x.T))
                func = -(mu_p / cov_p)
                return func

            # define equality constraint representing fully invested portfolio
            def constraintEq(x):
                A = np.ones(x.shape)
                b = 1
                constraintVal = np.matmul(A, x.T) - b
                return constraintVal

            # define bounds and other parameters
            xinit = np.repeat(1 / n, n)
            cons = ({'type': 'eq', 'fun': constraintEq})
            lb = 0
            ub = 1
            bnds = tuple([(lb, ub) for x in xinit])

            # invoke minimize solver
            res = minimize(f, x0=xinit, args=(_return, _cov), method='SLSQP',
                           bounds=bnds, constraints=cons, tol=10**-3)
            return res.x

        '===================================Portfolio Rebalance Part==================================================='
        # Exposure_controlling_process.
        def risk_mapping(VaR95_quantile, VaR99_quantile, CVaR95_quantile, CVaR99_quantile, VaR95, VaR99, CVaR95, CVaR99):
            """
            Calculate the scaling factors to adjust the exposures when huge VaR occurs.
            :param VaR95_quantile: predefined 95th VaR cutoff.
            :param VaR99_quantile: predefined 99th VaR cutoff.
            :param CVaR95_quantile: predefined 95th cVaR cutoff.
            :param CVaR99_quantile: predefined 99th cVaR cutoff.
            :param VaR95: the last period 95th VaR of our portfolio.
            :param VaR99: the last period 99th VaR of our portfolio.
            :param CVaR95: the last period 95th cVaR of our portfolio.
            :param CVaR99: the last period 99th cVaR of our portfolio.
            :return: a exposure scaling factor.
            """
            _scaling = 1
            indicator1, indicator2, indicator3, indicator4 = abs(VaR95) > abs(VaR95_quantile), abs(VaR99) > abs(VaR99_quantile), abs(CVaR95) > abs(CVaR95_quantile), abs(CVaR99) > abs(CVaR99_quantile)

            # rescale if any metric is higher than threshold.
            if indicator1:
                _scaling *= VaR95_quantile / VaR95
            if indicator2:
                _scaling *= VaR99_quantile / VaR99
            if indicator3:
                _scaling *= CVaR95_quantile / CVaR95
            if indicator4:
                _scaling *= CVaR99_quantile / CVaR99

            return abs(_scaling)

        # 50/50 for US and CA ETF
        USDcapital = initial_capital / 2
        CADcapital = initial_capital / 2
        capital = USDcapital + CADcapital

        # dataframes to store the portfolio info
        dollar_full_portfolio = pd.DataFrame()
        PnL = {}

        # initialize scaling factor
        _scalingfactor = 1

        # dataframes to store portfolio risk profiles
        USD_risk_profile = pd.DataFrame()
        CAD_risk_profile = pd.DataFrame()
        overall_risk_profile = pd.DataFrame()

        for i in range(1, len(self.semiannual.keys())):

            # Calculate the optimal weights
            
            # First period correlation is used to calculate optimal weights for next period.
            times_usd = self.semiannual[i - 1][top_cor[i][:cutoff]].index
            times_cad = self.semiannual[i - 1][top_cor[i][cutoff:]].index
            usd = self.semiannual[i - 1][top_cor[i][:cutoff]].subtract(riskfree['RFR'], axis=0)[times_usd[0]:times_usd[-1]]
            cad = self.semiannual[i - 1][top_cor[i][cutoff:]].subtract(riskfree['RFR'], axis=0)[times_cad[0]:times_cad[-1]].fillna(0)

            if benchmark is not None:
                benchmark_semi = benchmark[times_usd[0]:times_usd[-1]]
            else:
                benchmark_semi = None

            # used for scenario return adjustments
            if scenario == 'Market crash':
                sa_us = Risk_analytics.scenario_analysis(usd)
                usd = sa_us.all_crash()

                sa_ca = Risk_analytics.scenario_analysis(cad)
                cad = sa_ca.all_crash()
            elif scenario == 'Random ETF crash':
                sa_us = Risk_analytics.scenario_analysis(usd)
                usd = sa_us.random_crash()

                sa_ca = Risk_analytics.scenario_analysis(cad)
                cad = sa_ca.random_crash()
            elif scenario == 'MV ETF crash':
                sa_us = Risk_analytics.scenario_analysis(usd)
                usd = sa_us.mv_crash(8,0.75)
                sa_ca = Risk_analytics.scenario_analysis(cad)
                cad = sa_ca.mv_crash(8,0.75)

            # fit optimization based on specified type
            if optimization_type == 'MVO':
                if benchmark_semi is None:
                    usdweights = MVO(usd)
                    cadweights = MVO(cad)
                else:
                    usdweights = MVO(usd, benchmark_semi)
                    cadweights = MVO(cad, benchmark_semi)
            elif optimization_type == 'Risk parity':
                usdweights = Risk_parity(usd)
                cadweights = Risk_parity(cad)
            elif optimization_type == 'Sharpe ratio maximization':
                usdweights = sharpe_ratio_maximization(usd)
                cadweights = sharpe_ratio_maximization(cad)
            elif optimization_type == 'Equally weighted':
                usdweights = np.repeat(1 / cutoff, cutoff)
                cadweights = np.repeat(1 / cutoff, cutoff)
            else:
                raise Exception('Type not available.')

            # Calculate the returns
            usdreturns = self.semiannual[i][top_cor[i][:cutoff]]
            cadreturns = self.semiannual[i][top_cor[i][cutoff:]]

            '========================Monthly re-weighting and cash adjustments using risk metrics===================='
            # monthly separate the data
            dates_to_split = pd.date_range(usdreturns.index[0], usdreturns.index[-1], freq='W')
            monthly_usdreturns = {}
            monthly_cadreturns = {}
            for j in range(len(dates_to_split)-1):
                monthly_usdreturns[j] = usdreturns.loc[dates_to_split[j]:dates_to_split[j+1], :]
                monthly_cadreturns[j] = cadreturns.loc[dates_to_split[j]:dates_to_split[j+1], :]

            usdpnl = {}
            cadpnl = {}
            pnl = {}

            usd_portfolio_return = monthly_usdreturns[0].mul(usdweights, axis=1)
            cad_portfolio_return = monthly_cadreturns[0].mul(cadweights, axis=1)
            # remove outliers
            usd_portfolio_return[usd_portfolio_return.values > 100] = 0
            cad_portfolio_return[cad_portfolio_return.values > 100] = 0
            portfolio_return = pd.concat([usd_portfolio_return, cad_portfolio_return], axis=1)
            dollar_full_portfolio = pd.concat([dollar_full_portfolio, portfolio_return])
            usdpnl[0] = usd_portfolio_return.sum(axis=1) * USDcapital
            cadpnl[0] = cad_portfolio_return.sum(axis=1) * CADcapital
            pnl[0] = usdpnl[0] + cadpnl[0]
            capital += (pnl[0].cumsum()[-1] - pnl[0].cumsum()[0])
            USDcapital = capital/2
            CADcapital = capital/2

            # calculate risk metrics for each month of this sub-period.
            for j in range(1, len(monthly_usdreturns)):

                # A series of risk metrics referring back to the Risk_analytics module.
                _USDrisk = Risk_analytics.risk(monthly_usdreturns[j-1], usdweights, cutoff)
                _CADrisk = Risk_analytics.risk(monthly_cadreturns[j-1], cadweights, cutoff)
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
                _totalVaR95 = _USDVaR95 + _CADVaR95
                _totalVaR99 = _USDVaR95 + _CADVaR95
                _totalCVaR95 = _USDCVaR95 + _CADCVaR95
                _totalCVaR99 = _USDCVaR99 + _CADCVaR99

                # Scaling_factor: VaR95 cutoff, VaR 99 cutoff, CVaR95 cutoff, CVaR99 cutoff.
                _scalingfactor = risk_mapping(VaRcutoff['VaR95'], VaRcutoff['VaR99'], VaRcutoff['CVaR95'],
                                              VaRcutoff['CVaR99'], _totalVaR95, _totalVaR99, _totalCVaR95,
                                              _totalCVaR99)

                # end when reaching month end
                #if j == len(dates_to_split) - 1:
                #    break
                    
                # recalculate PnL using rescaled capital
                _USDcapital = USDcapital * _scalingfactor
                _CADcapital = CADcapital * _scalingfactor

                usd_portfolio_return = monthly_usdreturns[j].mul(usdweights, axis=1)
                cad_portfolio_return = monthly_cadreturns[j].mul(cadweights, axis=1)

                usd_portfolio_return[usd_portfolio_return.values > 100] = 0
                cad_portfolio_return[cad_portfolio_return.values > 100] = 0

                portfolio_return = pd.concat([usd_portfolio_return, cad_portfolio_return], axis=1)
                dollar_full_portfolio = pd.concat([dollar_full_portfolio, portfolio_return])

                # Partially PnL calculation
                usdpnl[j] = usd_portfolio_return.sum(axis=1) * _USDcapital
                cadpnl[j] = cad_portfolio_return.sum(axis=1) * _CADcapital
                pnl[j] = usdpnl[j]+cadpnl[j]

                # calculate the capital holdings at each period end
                USDcapital += (usdpnl[j].cumsum()[-1] - usdpnl[j].cumsum()[0])
                CADcapital += (cadpnl[j].cumsum()[-1] - cadpnl[j].cumsum()[0])
                capital += (pnl[j].cumsum()[-1] - pnl[j].cumsum()[0])

            # reconcil the PnL 50/50 at each semi annual start
            # Note: we don't need to do rebalancing operation. Just reallocate the capital is enough.
            PnL[i] = pd.concat(pnl)
            # Add capital for every half year
            if i% 6:
                capital += 10000
            USDcapital = capital / 2
            CADcapital = capital / 2

            '================================Semi-annual risk metrics access==========================================='
            # Calculate the overall risk
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
            _totalVaR95 = _USDVaR95 + _CADVaR95
            _totalVaR99 = _USDVaR95 + _CADVaR95
            _totalCVaR95 = _USDCVaR95 + _CADCVaR95
            _totalCVaR99 = _USDCVaR99 + _CADCVaR99

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

            # store risk metrics into dataframes
            _USDriskprofile = pd.DataFrame.from_dict(_USDriskprofile)
            _CADriskprofile = pd.DataFrame.from_dict(_CADriskprofile)
            _overall_risk_profile = pd.DataFrame.from_dict(_overallriskprofile)

            USD_risk_profile = pd.concat([USD_risk_profile, _USDriskprofile])
            CAD_risk_profile = pd.concat([CAD_risk_profile, _CADriskprofile])
            overall_risk_profile = pd.concat([overall_risk_profile, _overall_risk_profile])

        PnL = pd.concat(PnL).droplevel(level=[0, 1])

        USD_risk_profile.set_index('Period', inplace=True)
        CAD_risk_profile.set_index('Period', inplace=True)
        overall_risk_profile.set_index('Period', inplace=True)

        # max drawdown calculation:
        ts = PnL.cumsum()
        previous_peaks = ts.cummax()
        drawdown = (ts - previous_peaks) / previous_peaks

        max_drawdown = f'{drawdown.min()*100}%'
        # Sharpe ratio
        _sharpe = (PnL/initial_capital).subtract(riskfree['RFR'],axis=0).mean() *initial_capital/PnL.std() * np.sqrt(PnL.shape[0])
        
        # Final weights
        _dict = {}
        for index,etf in enumerate(top_cor[len(top_cor)]):
            if index < cutoff:
                _dict[etf] = usdweights[index]
            else:
                _dict[etf] = cadweights[index-cutoff]
        _finalweights = pd.DataFrame.from_dict(_dict,orient='index')
        _finalweights.rename(columns={0: "final_weights"},inplace = True)

        return dollar_full_portfolio.fillna(0),PnL, USD_risk_profile, CAD_risk_profile, overall_risk_profile, max_drawdown,_sharpe,_finalweights
