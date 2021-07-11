import math
import numpy as np

from scipy.optimize import linprog, minimize
from scipy.stats.distributions import chi2


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

    def __init__(self, returns, factReturns, lam=0.01):
        """
        Initialize the portfolio optimization.
        :param returns: dataframe that contains the returns of all our assets.
        :param factReturns: dataframe that contains all factorial returns.
        :param lam: lambda parameter for LASSO, default = 0.01.
        """
        self.returns = returns.to_numpy()
        self.factReturns = factReturns.to_numpy()

        # implement Ordinary Least Square (OLS)
        [T, p] = self.factReturns.shape
        X = np.concatenate((np.ones((T, 1)), self.factReturns), axis=1)
        B = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(self.returns)
        a = B[0, :].T
        V = B[1:, :]

        # residual variance
        ep = self.returns - X.dot(B)
        sigma_ep = 1 / ((T - p - 1) * np.sum(np.square(ep), 0))
        D = np.diag(sigma_ep)

        f_bar = np.mean(self.factReturns, 0).T
        F = np.cov(self.factReturns, rowvar=False)

        self.mu = a + V.T.dot(f_bar)

        Q = (V.T.dot(F)).dot(V) + D
        self.Q = np.nan_to_num(Q, copy=True, posinf=0, neginf=0)

    def MVO(self):
        """
        Perform Mean Variance Optimization.
        :return: weights of assets
        """
        # total number of assets
        n = self.Q.shape[0]
        # target return = average return of all assets
        targetRet = np.mean(self.mu)

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
                {'type': 'ineq', 'fun': constraintIneq, 'args': (self.mu, targetRet)})
        lb = 0
        ub = 1
        bnds = tuple([(lb, ub) for x in xinit])

        res = minimize(f, args=(self.Q), method='trust-constr',
                       x0=xinit, bounds=bnds, constraints=cons, tol=10 ** -3)

        return res.x

    def robust_MVO(self, alpha=0.95, lam=0.02):
        """
        Perform robust MVO.
        :param alpha:
        :param lam:
        :return: weights of assets
        """
        # min   lambda * (x' * Q * x) - mu' x + epsilon * norm (sqrtTh * x)
        # s.t.  sum(x) == 1
        #       x >= 0

        # number of assets
        n = self.Q.shape[0]
        # number of return observations
        T = self.returns.shape[0]

        # radius of uncertainty set
        ep = math.sqrt(chi2.ppf(alpha, n))

        # theta is squared standard error of our expected returns
        theta = np.diag(np.diag(self.Q)) / T

        sqrtTh = np.sqrt(theta)

        # parameters for nonlinear program
        def f(x, mu, Q, lam, sqrtTh, ep):
            func = lam * np.matmul(np.matmul(x, Q), x.T) - np.matmul(np.array(mu), x.T) + ep * np.linalg.norm(
                np.matmul(sqrtTh, x), axis=0)
            return func

        def constraintEq(x):
            Aeq = np.ones(x.shape)
            beq = 1
            EqconstraintVal = np.matmul(Aeq, x.T) - beq
            return EqconstraintVal

        xinit = np.repeat(1 / n, n)
        cons = ({'type': 'eq', 'fun': constraintEq})
        lb = 0
        ub = 1
        bnds = tuple([(lb, ub) for x in xinit])

        res = minimize(f, args=(self.mu, self.Q, lam, sqrtTh, ep), method='trust-constr',
                       x0=xinit, bounds=bnds, constraints=cons, tol=10 ** -3)

        return res.x

    def Risk_parity(self, kappa=5):
        """
        Perform Risk Parity.
        :param kappa:
        :return: weights of assets
        """
        # min   (1/2) * (x' * Q * x) - kappa * sum_{i=1}^n ln(x_i)
        # s.t.  x >= 0

        # number of assets
        n = self.Q.shape[0]

        # parameters for nonlinear program

        eps = 1e-3

        def safe_log(x):
            if x > eps:
                return math.log(x)
            logeps = math.log(eps)
            a = 1 / (3 * eps * (3 * logeps * eps) ** 2)
            b = eps * (1 - 3 * logeps)
            return a * (x - b) ** 3

        def f(x, mu, Q, kappa):
            func = 0.5 * np.matmul(np.matmul(x, Q), x.T)

            for i in range(n):
                func = func - kappa * safe_log(x[i])

            return func

        xinit = np.repeat(1 / n, n)
        lb = 0
        ub = 1
        bnds = tuple([(lb, ub) for x in xinit])

        res = minimize(f, args=(self.mu, self.Q, kappa), method='trust-constr',
                       x0=xinit, bounds=bnds, tol=10 ** -3)

        return res.x / np.sum(res.x)

    def CVaR_optimization(self, alpha=0.95):
        """
        Perform CVaR optimization
        :param alpha:
        :return: weights of assets
        """
        # target return
        R = np.mean(self.mu)

        # number of assets and scenarios
        [S, n] = self.returns.shape

        # min     gamma + (1 / [(1 - alpha) * S]) * sum( z_s )
        # s.t.    z_s   >= 0,                 for s = 1, ..., S
        #         z_s   >= -r_s' x - gamma,   for s = 1, ..., S
        #         1' x  =  1,
        #         mu' x >= R

        lb = np.concatenate((np.zeros((n, 1)), np.zeros((S, 1)), [[0]]), axis=0)
        bounds = []
        for lbs in lb:
            bound = [lbs[0], None]
            bounds.append(bound)

        # nonlinear constraints
        A_1 = np.concatenate((-1 * self.returns, -np.eye(S), -1 * np.ones((S, 1))), axis=1)
        A_2 = np.concatenate(([-1 * self.mu.T], np.zeros((1, S)), [[0]]), axis=1)
        A = np.concatenate((A_1, A_2), axis=0)
        b = np.concatenate((np.zeros((S, 1)), [[-1 * R]]), axis=0)

        # equality constraints
        Aeq = np.concatenate((np.ones((1, n)), np.zeros((1, S)), [[0]]), axis=1)
        beq = 1

        # objective linear cost function
        k = (1 / ((1 - alpha) * S))
        c = np.concatenate((np.zeros((n, 1)), k * np.ones((S, 1)), [[1]]), axis=0)

        y = linprog(c, A, b, Aeq, beq, bounds=bounds, method='interior-point')

        return y.x[:n]
