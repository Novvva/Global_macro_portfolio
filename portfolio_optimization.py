import numpy as np
import math

from scipy.optimize import linprog, minimize
from qpsolvers import solve_qp
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

    def __init__(self, returns, factReturns, model_type, lam=0.01):
        """
        Initialize the portfolio optimization.
        :param returns: dataframe that contains the returns of all our assets.
        :param factReturns: dataframe that contains all factorial returns.
        :param model_type: regression model type to find Q and mu.
        :param lam: lambda parameter for LASSO, default = 0.01.
        """
        self.returns = returns.to_numpy()
        self.factReturns = factReturns.to_numpy()

        if model_type == 'OLS':
            # implement Ordinary Least Square
            x = factReturns.copy()

            # Add the constant line
            X = x.insert(0, 'constant', np.ones(1))
            # Optimize the residuals
            B = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(returns)
            # Calculate the residuals
            e = returns.values - X.T.dot(B)
            # Calculate the variance of residuals
            ve = e.T.dot(e) / (X.shape[0] - X.shape[1] - 1)
            # The vector of expected asset return
            f = factReturns.mean()
            _f = f.copy().insert(0, 'constant', np.ones(1))
            # Calculate the Factor Covariance matrix
            F = factReturns.cov()

            Q = np.delete(B, 0, axis=1).T.dot(F).dot(np.delete(B, 0, axis=1)) + np.diag(np.diag(ve))
            self.Q = (Q + Q.T) / 2
            self.mu = B.T.dot(_f)

        elif model_type == 'LASSO':
            # implement LASSO regression
            x = factReturns.to_numpy()
            r = returns.to_numpy()
            [T, p] = x.shape
            n = r.shape[1]

            # placeholder for coefficients
            B = np.ones((p+1, n))
            X = np.concatenate((np.ones((T, 1)), x), axis=1)
            for i in range(n):
                ret = r[:, i]

                X_au = np.concatenate((X, np.zeros((T, p+1))), axis=1)

                # min 2 B 'X_au' X_au B - 2r_i'X_au B + auxiliary terms * B
                # s.t. y >= B_i (B_i - y <= 0)
                #      y >= -B_i (-B_i - y <= 0)
                Q = 2 * X_au.T.dot(X_au)
                auxiliary = np.concatenate((np.zeros((1, p+1)), lam*np.ones((1, p+1))), axis=1)
                c = (-2 * ret.T.dot(X_au) + auxiliary).T

                A_1 = np.concatenate((np.eye(p+1), -1 * np.eye(p+1)), axis=1)
                A_2 = np.concatenate((-1 * np.eye(p+1), -1 * np.eye(p+1)), axis=1)
                A = np.concatenate((A_1, A_2), axis=0)
                b = np.zeros((np.shape(A)[0], 1))

                res = solve_qp(Q, c, A, b, [], [], [], [])
                # round too small coefficient to 0
                B[:, i] = np.round(res[:p+2], 4)

            V = B[1:p+2, :]
            # Calculate the residuals
            e = r - X.dot(B)
            D = np.zeros((1, e.shape[1]))
            for j in range(e.shape[1]):
                D[1, j] = (np.linalg.norm(e[:, j]) ** 2) / (T - p - 1)
            D = np.diag(D)

            Q = (V.T.dot(np.cov(x))).dot(V) + D
            self.Q = (Q + Q.T) / 2
            self.mu = B.T.dot(np.mean(X, axis=0))

        else:
            raise Exception('model type is not defined!')

    def MVO(self):
        """
        Perform Mean Variance Optimization.
        :return: weights of assets
        """
        # total number of assets
        n = self.Q.shape[0]
        # target return = average return of all assets
        targetRet = np.mean(self.mu)

        # disallow short sales
        lb = np.zeros((n, 1))

        # expected return constraint
        A = -1 * self.mu.T
        b = -1 * targetRet

        # constraint weights to sum to 1
        Aeq = np.ones((1, n))
        beq = 1

        return solve_qp(2 * self.Q, [], A, b, Aeq, beq, lb, [])

    def robust_MVO(self, alpha, lam):
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
        sqrtTh = math.sqrt(theta)

        # parameters for nonlinear program
        # initial portfolio - equally weighted
        x0 = np.tile(1/n, (n, 1))
        con = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bnds = ((0, None),) * n

        res = minimize(lambda x: (lam * ((x.T.dot(self.Q)).dot(x))) - self.mu.T.dot(x) + ep * np.linalg.norm(sqrtTh * x),
                       x0, method='SLSQP', bounds=bnds, constraints=con)
        return res.x

    def Risk_parity(self, kappa):
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
        # initial portfolio - equally weighted
        x0 = np.tile(1/n, (n, 1))
        bnds = ((0, None),) * n

        res = minimize(lambda x: 0.5 * ((x.T.dot(self.Q)).dot(x)) - kappa * np.sum(np.log(x)),
                       x0, method='SLSQP', bounds=bnds)

        return res.x

    def CVaR_optimization(self, alpha):
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
        A_2 = np.concatenate((-1 * self.mu.T, np.zeros((1, S)), [[0]]), axis=1)
        A = np.concatenate((A_1, A_2), axis=0)
        b = np.concatenate((np.zeros((S, 1)), -1 * R), axis=0)

        # equality constraints
        Aeq = np.concatenate((np.ones(1, n), np.zeros(1, S), [[0]]), axis=1)
        beq = 1

        # objective linear cost function
        k = (1 / ((1-alpha) * S))
        c = np.concatenate((np.zeros((n, 1)), k * np.ones((S, 1)), [[1]]), axis=0)

        y = linprog(c, A, b, Aeq, beq, bounds=bounds, method='interior-point')

        return y.x[:n+1]
