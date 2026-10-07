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
