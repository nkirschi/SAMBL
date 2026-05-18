"""
Agents for episodic continuous-time LQ control.
All learning agents use the Riccati ODE for planning.
"""

import numpy as np
from planner import RiccatiODESolver
from estimator import DiscreteRidgeEstimator, RowLassoEstimator


class Agent:
    """Base class for all agents."""

    def __init__(self, config, Q, R):
        self.config = config
        self.Q = Q
        self.R = R
        self.planner = RiccatiODESolver(config, Q, R)
        self.A_est = None
        self.B_est = None

        # Previous gain (for fallback on DRE failure)
        self._prev_dre_solution = None
        self._prev_B_est = None
        self._dre_valid = False

    def get_control(self, t, x):
        raise NotImplementedError

    def update(self, buffer):
        raise NotImplementedError

    def _solve_dre(self, A, B):
        """Solve DRE. In case of failure fall back to previous solution."""
        try:
            sol = self.planner.solve(A, B)
            if not sol.success:
                raise RuntimeError(f"DRE solver did not converge: {sol.message}")
            self._prev_dre_solution = sol
            self._prev_B_est = B.copy()
            self._dre_valid = True
            return True
        except Exception:
            if self._prev_dre_solution is not None:
                self.planner.solution = self._prev_dre_solution
            self._dre_valid = False
            return False

    def _get_feedback(self, t, x):
        B = self._prev_B_est if not self._dre_valid else self.B_est
        if B is None:
            return np.zeros(self.config.u_dim)
        K = self.planner.get_K(t, B)
        return K @ x


class OracleAgent(Agent):
    """Uses true (A_true, B_true); solves DRE once at construction."""

    def __init__(self, config, Q, R, A_true, B_true):
        super().__init__(config, Q, R)
        self.A_est = A_true.copy()
        self.B_est = B_true.copy()
        self._solve_dre(A_true, B_true)

    def get_control(self, t, x):
        return self._get_feedback(t, x)

    def update(self, buffer):
        pass


class DenseGreedyAgent(Agent):
    """Ridge regression estimator + DRE feedback. No excitation."""

    def __init__(self, config, Q, R, A_init, B_init, mu=0.01):
        super().__init__(config, Q, R)
        self.estimator = DiscreteRidgeEstimator(config.x_dim, config.u_dim, mu=mu)
        self.A_est = A_init.copy()
        self.B_est = B_init.copy()
        self._solve_dre(A_init, B_init)

    def get_control(self, t, x):
        return self._get_feedback(t, x)

    def update(self, buffer):
        self.A_est, self.B_est = self.estimator.estimate(buffer)
        self._solve_dre(self.A_est, self.B_est)


class DenseExcitedAgent(Agent):
    """Ridge regression estimator + DRE feedback + additive excitation."""

    def __init__(
        self,
        config,
        Q,
        R,
        A_init,
        B_init,
        sigma_u=0.1,
        excitation_rng=None,
        mu=0.01,
    ):
        super().__init__(config, Q, R)
        self.estimator = DiscreteRidgeEstimator(config.x_dim, config.u_dim, mu=mu)
        self.sigma_u = sigma_u
        self._exc_rng = excitation_rng or np.random.RandomState(0)
        self.A_est = A_init.copy()
        self.B_est = B_init.copy()
        self._solve_dre(A_init, B_init)

    def get_control(self, t, x):
        return (
            self._get_feedback(t, x)
            + self._exc_rng.randn(self.config.u_dim) * self.sigma_u
        )

    def update(self, buffer):
        self.A_est, self.B_est = self.estimator.estimate(buffer)
        self._solve_dre(self.A_est, self.B_est)


class SparseGreedyAgent(Agent):
    """Row-wise Lasso estimator + DRE feedback. No excitation."""

    def __init__(
        self,
        config,
        Q,
        R,
        A_init,
        B_init,
        lambda_fixed=None,
        sigma_bar=None,
        c_lambda=2.0,
        delta=0.05,
        max_episodes=None,
        max_iter=5000,
        tol=1e-4,
    ):
        super().__init__(config, Q, R)
        self.estimator = RowLassoEstimator(
            x_dim=config.x_dim,
            u_dim=config.u_dim,
            lambda_fixed=lambda_fixed,
            sigma_bar=sigma_bar,
            c_lambda=c_lambda,
            delta=delta,
            max_episodes=max_episodes,
            max_iter=max_iter,
            tol=tol,
        )
        self.A_est = A_init.copy()
        self.B_est = B_init.copy()
        self._solve_dre(A_init, B_init)

    def get_control(self, t, x):
        return self._get_feedback(t, x)

    def update(self, buffer):
        self.A_est, self.B_est = self.estimator.estimate(buffer)
        self._solve_dre(self.A_est, self.B_est)


class SparseExcitedAgent(Agent):
    """Row-wise Lasso estimator + DRE feedback + additive excitation."""

    def __init__(
        self,
        config,
        Q,
        R,
        A_init,
        B_init,
        sigma_u=0.1,
        excitation_rng=None,
        lambda_fixed=None,
        sigma_bar=None,
        c_lambda=2.0,
        delta=0.05,
        max_episodes=None,
        max_iter=5000,
        tol=1e-4,
    ):
        super().__init__(config, Q, R)
        self.estimator = RowLassoEstimator(
            x_dim=config.x_dim,
            u_dim=config.u_dim,
            lambda_fixed=lambda_fixed,
            sigma_bar=sigma_bar,
            c_lambda=c_lambda,
            delta=delta,
            max_episodes=max_episodes,
            max_iter=max_iter,
            tol=tol,
        )
        self.sigma_u = sigma_u
        self._exc_rng = excitation_rng or np.random.RandomState(0)
        self.A_est = A_init.copy()
        self.B_est = B_init.copy()
        self._solve_dre(A_init, B_init)

    def get_control(self, t, x):
        return (
            self._get_feedback(t, x)
            + self._exc_rng.randn(self.config.u_dim) * self.sigma_u
        )

    def update(self, buffer):
        self.A_est, self.B_est = self.estimator.estimate(buffer)
        self._solve_dre(self.A_est, self.B_est)
