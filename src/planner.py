from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve
from common import SystemConfig


class RiccatiODESolver:
    """
    Solves the matrix Riccati DIFFERENTIAL equation backwards on a uniform grid.
    Not to be confused with the algebraic Riccati equation.
    See Basei et al. 2022 equation (2.10).

    In the tau = T - t parametrisation the RDE integrates forward from P(0) = 0:
        dP/dtau = A^T P + P A - P S P + Q,   S = B R^{-1} B^T.
    We integrate with explicit DOP853 (Jacobian-free, so no d^2 x d^2 factorisation)
    and evaluate P on the control grid tau_j = j*dt (j = 0..H), which is all get_K
    ever queries. A benchmark across d in {5,10,20,50,100} found DOP853 the fastest
    correct option -- ~6x faster than the Hamiltonian matrix exponential at d=100,
    at ~1e-9 accuracy (the gain K = -R^{-1} B^T P is insensitive well below that).
    Radau is kept as a stiff-case fallback should the explicit solver fail.
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
        grid = self._integrate(A, S, terminal_cost, "DOP853")
        if grid is None:
            grid = self._integrate(A, S, terminal_cost, "Radau")  # stiff fallback
        if grid is None:
            return False
        self._P_grid = grid
        return True

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
