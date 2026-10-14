"""Univariate linear regression implemented with NumPy.

Two solvers are provided so results can be cross-checked:
  * normal equation (closed form)
  * batch gradient descent (with feature scaling for stable convergence)
"""
from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, method: str = "normal", lr: float = 0.1, n_iter: int = 2000):
        if method not in {"normal", "gd"}:
            raise ValueError("method must be 'normal' or 'gd'")
        self.method, self.lr, self.n_iter = method, lr, n_iter
        self.intercept_: float = 0.0
        self.coef_: float = 0.0
        self.cost_history_: list[float] = []

    @staticmethod
    def _cost(x, y, b0, b1) -> float:
        return float(np.mean((b0 + b1 * x - y) ** 2) / 2)

    def fit(self, x, y) -> "LinearRegression":
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.shape != y.shape or x.size < 2:
            raise ValueError("x and y must have the same length (>= 2)")

        if self.method == "normal":
            X = np.column_stack([np.ones_like(x), x])
            theta = np.linalg.solve(X.T @ X, X.T @ y)
            self.intercept_, self.coef_ = float(theta[0]), float(theta[1])
            self.cost_history_ = [self._cost(x, y, *theta)]
            return self

        # Gradient descent on standardized x, then map back to original units.
        mu, sigma = x.mean(), x.std() or 1.0
        z = (x - mu) / sigma
        b0, b1, hist = 0.0, 0.0, []
        for _ in range(self.n_iter):
            err = b0 + b1 * z - y
            b0 -= self.lr * err.mean()
            b1 -= self.lr * (err * z).mean()
            hist.append(self._cost(z, y, b0, b1))
        self.coef_ = float(b1 / sigma)
        self.intercept_ = float(b0 - b1 * mu / sigma)
        self.cost_history_ = hist
        return self

    def predict(self, x) -> np.ndarray:
        return self.intercept_ + self.coef_ * np.asarray(x, dtype=float)

    def score(self, x, y) -> float:
        """R^2 coefficient of determination."""
        y = np.asarray(y, dtype=float)
        ss_res = np.sum((y - self.predict(x)) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot)


class LogLogRegression:
    """log(y) = a + b * log(x): a power law y ~ x**b, fitted by least squares on logs.

    Why: when spread grows with city size (as with real sales data), errors are
    proportional, not constant. Working in logs makes the spread roughly constant
    and makes the model minimise *percentage* error. ``coef_`` is the elasticity:
    a 1% larger population goes with a ``coef_``% larger y.

    ``predict`` returns the expected value of y (log-scale prediction times
    Duan's smearing factor, which corrects the bias from back-transforming).
    """

    def __init__(self):
        self.linear_ = LinearRegression("normal")
        self.intercept_: float = 0.0     # a, on the log scale
        self.coef_: float = 0.0          # b, the elasticity
        self.smearing_: float = 1.0

    @staticmethod
    def _logs(x, y=None):
        x = np.asarray(x, dtype=float).ravel()
        if np.any(x <= 0):
            raise ValueError("log model needs strictly positive population")
        if y is None:
            return np.log(x)
        y = np.asarray(y, dtype=float).ravel()
        if np.any(y <= 0):
            raise ValueError("log model needs strictly positive target values")
        return np.log(x), np.log(y)

    def fit(self, x, y) -> "LogLogRegression":
        lx, ly = self._logs(x, y)
        self.linear_.fit(lx, ly)
        self.intercept_, self.coef_ = self.linear_.intercept_, self.linear_.coef_
        self.smearing_ = float(np.mean(np.exp(ly - self.linear_.predict(lx))))
        return self

    def predict_log(self, x) -> np.ndarray:
        return self.linear_.predict(self._logs(x))

    def predict(self, x) -> np.ndarray:
        return self.smearing_ * np.exp(self.predict_log(x))

    def score(self, x, y) -> float:
        """R^2 in the original units (comparable with LinearRegression.score)."""
        y = np.asarray(y, dtype=float).ravel()
        ss_res = np.sum((y - self.predict(x)) ** 2)
        return float(1 - ss_res / np.sum((y - y.mean()) ** 2))
