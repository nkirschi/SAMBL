from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Lasso
from common import EstimatorConfig, SystemConfig


class RegressionBuffer:
    """Pre-allocated buffer for discrete regression samples."""

    def __init__(
        self, x_dim: int, u_dim: int, max_episodes: int, steps_per_episode: int
    ):
        self.x_dim = x_dim
        self.z_dim = x_dim + u_dim
        self.capacity = max_episodes * steps_per_episode
        self._Z = np.zeros((self.capacity, self.z_dim), dtype=np.float64)
        self._Y = np.zeros((self.capacity, x_dim), dtype=np.float64)
        self._N = 0

    @property
    def N(self) -> int:
        return self._N

    @property
    def Z(self) -> NDArray[np.float64]:
        return self._Z[: self._N]

    @property
    def Y(self) -> NDArray[np.float64]:
        return self._Y[: self._N]

    def add_episode(self, zs: NDArray[np.float64], ys: NDArray[np.float64]) -> None:
        H = zs.shape[0]
        start = self._N
        end = start + H
        if end > self.capacity:
            raise RuntimeError("Buffer overflow. Increase max_episodes.")
        self._Z[start:end] = zs
        self._Y[start:end] = ys
        self._N = end


class DiscreteRidgeEstimator:
    """
    Discrete-time ridge regression estimator (dense baseline).
    """

    def __init__(self, est_cfg: EstimatorConfig, lamda_schedule: callable = None):
        self.lamda_fixed = est_cfg.mu_ridge
        self.lamda_schedule = lamda_schedule

    def fit(
        self, z_data: NDArray[np.float64], y_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n_samples, n_features = z_data.shape
        lamda = (
            self.lamda_schedule(n_samples)
            if self.lamda_fixed is None
            else self.lamda_fixed
        )
        gram = (z_data.T @ z_data) / max(n_samples, 1)
        rhs = (z_data.T @ y_data) / max(n_samples, 1)
        lhs = gram + lamda * np.eye(n_features)
        try:
            theta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return theta.T.astype(np.float64)


class RowLassoEstimator:
    """
    Row-wise Lasso estimator for row-sparse parameters.
    """

    def __init__(
        self,
        sys_cfg: SystemConfig,
        est_cfg: EstimatorConfig,
        lamda_schedule: callable = None,
    ):
        self.x_dim = sys_cfg.d
        self.z_dim = sys_cfg.d + sys_cfg.p
        self.lamda_fixed = est_cfg.lambda_lasso
        self.lamda_schedule = lamda_schedule

        # Instantiate sklearn models ONCE to bypass python loop overhead
        self.models = [
            Lasso(
                fit_intercept=False,
                warm_start=True,
                max_iter=est_cfg.lasso_max_iter,
                tol=est_cfg.lasso_tol,
            )
            for _ in range(self.x_dim)
        ]

    def fit(
        self, z_data: NDArray[np.float64], y_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        theta_hat = np.zeros((self.x_dim, self.z_dim), dtype=np.float64)
        n_samples, _ = z_data.shape
        lamda = (
            self.lamda_schedule(n_samples)
            if self.lamda_fixed is None
            else self.lamda_fixed
        )

        for i in range(self.x_dim):
            self.models[i].alpha = lamda
            self.models[i].fit(z_data, y_data[:, i])
            theta_hat[i, :] = self.models[i].coef_

        return theta_hat
