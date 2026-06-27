from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve, expm
from common import SystemConfig


class RiccatiODESolver:
    """
    Solves the matrix Riccati DIFFERENTIAL equation backwards on a uniform grid.
    Not to be confused with the algebraic Riccati equation.
    See Basei et al. 2022 equation (2.10).

    In the tau = T - t parametrisation the RDE integrates forward from P(0) = 0:
        dP/dtau = A^T P + P A - P S P + Q,   S = B R^{-1} B^T.
    """

    _RTOL = 1e-8
    _ATOL = 1e-10

    def __init__(self, sys_cfg: SystemConfig, Q: NDArray[np.float64], R: NDArray[np.float64]):
        self.sys_cfg = sys_cfg
        self.Q = Q
        self.R = R
        self.R_cho = cho_factor(R)
        self._P_grid: NDArray[np.float64] | None = None  # (H+1, d, d), P at tau=j*dt

    def solve(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        terminal_cost: NDArray[np.float64] | None = None,
    ) -> bool:
        S = B @ cho_solve(self.R_cho, B.T)
        grid = self._integrate_expm(A, S, terminal_cost)
        if grid is None:  # safety net for a degenerate Mobius step
            grid = self._integrate(A, S, terminal_cost, "DOP853")
        if grid is None:
            grid = self._integrate(A, S, terminal_cost, "Radau")
        if grid is None:
            return False
        self._P_grid = grid
        return True

    def _integrate_expm(
        self,
        A: NDArray[np.float64],
        S: NDArray[np.float64],
        terminal_cost: NDArray[np.float64] | None,
    ) -> NDArray[np.float64] | None:
        d, H, dt = self.sys_cfg.d, self.sys_cfg.H, self.sys_cfg.dt
        M = np.block([[-A, S], [self.Q, A.T]])
        E = expm(M * dt)
        if not np.all(np.isfinite(E)):
            return None
        E11, E12, E21, E22 = E[:d, :d], E[:d, d:], E[d:, :d], E[d:, d:]
        grid = np.empty((H + 1, d, d), dtype=np.float64)
        P = np.zeros((d, d)) if terminal_cost is None else np.asarray(terminal_cost, float)
        grid[0] = 0.5 * (P + P.T)
        for k in range(1, H + 1):
            num = E21 + E22 @ P
            den = E11 + E12 @ P
            try:
                P = np.linalg.solve(den.T, num.T).T  # P = num @ den^{-1}
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(P)):
                return None
            grid[k] = 0.5 * (P + P.T)
        return grid

    def _integrate(
        self,
        A: NDArray[np.float64],
        S: NDArray[np.float64],
        terminal_cost: NDArray[np.float64] | None,
        method: str,
    ) -> NDArray[np.float64] | None:
        d, H, dt, T = self.sys_cfg.d, self.sys_cfg.H, self.sys_cfg.dt, self.sys_cfg.T

        def rhs(_, p_flat):
            P = p_flat.reshape(d, d)
            return (A.T @ P + P @ A - P @ S @ P + self.Q).ravel()

        p0 = np.zeros(d * d) if terminal_cost is None else terminal_cost.ravel()
        sol = solve_ivp(
            rhs, [0, T], p0, method=method,
            rtol=self._RTOL, atol=self._ATOL, dense_output=True,
        )
        if not sol.success:
            return None
        # Materialise P on the control grid and discard the dense interpolant.
        # get_K only ever queries grid points (t = k*dt), and is called every
        # control step, so a precomputed grid + cheap lerp beats re-evaluating
        # scipy's dense output (~3x faster per call) and keeps get_K independent
        # of which integrator ran.
        grid = np.empty((H + 1, d, d), dtype=np.float64)
        for j in range(H + 1):
            P = sol.sol(j * dt).reshape(d, d)
            grid[j] = 0.5 * (P + P.T)
        return grid

    def get_K(self, t: float, B: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Returns feedback matrix K(t) = -R^-1 B^T P(t).
        P(t) is read from the grid at tau = T - t (linear interpolation between
        the two bracketing grid points; exact at grid points).
        """
        if self._P_grid is None:
            return np.zeros((self.sys_cfg.p, self.sys_cfg.d), dtype=np.float64)

        H, dt = self.sys_cfg.H, self.sys_cfg.dt
        tau = self.sys_cfg.T - t  # map physics time t -> solver time tau
        pos = np.clip(tau / dt, 0.0, H)
        lo = int(np.floor(pos))
        hi = min(lo + 1, H)
        w = pos - lo
        P = (1.0 - w) * self._P_grid[lo] + w * self._P_grid[hi]
        return cho_solve(self.R_cho, -B.T @ P)
