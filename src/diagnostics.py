"""
Diagnostics for probing the theoretical assumptions.

Each function takes readily available data and returns a scalar or small array.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Set, List


def relative_parameter_error(
    theta_hat: NDArray[np.float64], theta_true: NDArray[np.float64]
) -> dict[str, float]:
    """
    Relative Frobenius error, reported jointly and for A, B blocks separately.
    """
    d = theta_true.shape[0]
    a_hat, b_hat = theta_hat[:, :d], theta_hat[:, d:]
    a_true, b_true = theta_true[:, :d], theta_true[:, d:]

    def _rel_err(est, true):
        denom = max(float(np.linalg.norm(true, "fro")), 1e-15)
        return float(np.linalg.norm(est - true, "fro") / denom)

    return {
        "joint": _rel_err(theta_hat, theta_true),
        "A": _rel_err(a_hat, a_true),
        "B": _rel_err(b_hat, b_true),
    }


def support_metrics(
    theta_hat: NDArray[np.float64], true_supports: List[Set[int]], threshold: float
) -> dict[str, dict[str, float]]:
    """
    Support precision, recall, and F1 averaged over rows, reported
    jointly and for the A and B blocks separately.
    """
    d = theta_hat.shape[0]
    joint_p, joint_r = [], []
    a_p, a_r = [], []
    b_p, b_r = [], []

    def _row_metrics(est_sup, true_sup):
        tp = len(est_sup & true_sup)
        return tp / max(len(est_sup), 1), tp / max(len(true_sup), 1)

    for i in range(d):
        est_full = set(np.where(np.abs(theta_hat[i]) > threshold)[0].tolist())
        true_full = true_supports[i]

        est_a, true_a = {j for j in est_full if j < d}, {j for j in true_full if j < d}
        est_b, true_b = (
            {j for j in est_full if j >= d},
            {j for j in true_full if j >= d},
        )

        p_j, r_j = _row_metrics(est_full, true_full)
        joint_p.append(p_j)
        joint_r.append(r_j)

        if true_a or est_a:
            p_a, r_a = _row_metrics(est_a, true_a)
            a_p.append(p_a)
            a_r.append(r_a)
        if true_b or est_b:
            p_b, r_b = _row_metrics(est_b, true_b)
            b_p.append(p_b)
            b_r.append(r_b)

    def _aggregate(p_list, r_list):
        if not p_list:
            return {"precision": np.nan, "recall": np.nan, "f1": np.nan}
        prec = float(np.mean(p_list))
        rec = float(np.mean(r_list))
        f1 = 2 * prec * rec / max(prec + rec, 1e-15)
        return {"precision": prec, "recall": rec, "f1": f1}

    return {
        "joint": _aggregate(joint_p, joint_r),
        "A": _aggregate(a_p, a_r),
        "B": _aggregate(b_p, b_r),
    }


# TODO: consider sampling from cones and reporting min Rayleigh coefficient
def restricted_gram_min_eigenvalue(
    Z: NDArray[np.float64], true_supports: List[Set[int]]
) -> float:
    """
    Minimum eigenvalue of the empirical gram matrix restricted to each row's
    true support, then the minimum over rows.

    This is an upper bound on the restricted eigenvalue constant kappa.
    Hence, it can be used as a necessary condition for Lasso recovery.
    """
    N = Z.shape[0]
    if N == 0:
        return 0.0

    min_eig = np.inf
    for supp in true_supports:
        cols = sorted(supp)
        if not cols:
            continue
        Z_S = Z[:, cols]
        gram = (Z_S.T @ Z_S) / N
        min_eig = min(min_eig, float(np.min(np.linalg.eigvalsh((gram + gram.T) / 2.0))))

    return min_eig if np.isfinite(min_eig) else 0.0


def regressor_energy_bound(Z):
    """
    Maximum average column energy.

    This is B from Proposition 10 (bounded regressor energy).
    """
    N = Z.shape[0]
    if N == 0:
        return 0.0
    return np.max(np.linalg.norm(Z, axis=0) / np.sqrt(N))


def closed_loop_spectral_abscissa(
    A_true: NDArray[np.float64], B_true: NDArray[np.float64], K: NDArray[np.float64]
) -> float:
    """
    Maximum real part of eigenvalues of the closed-loop matrix A + BK.
    """
    return float(np.max(np.real(np.linalg.eigvals(A_true + B_true @ K))))


def episode_cost(
    X: NDArray[np.float64],
    U: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
    dt: float,
) -> float:
    """
    Compute one episode's quadratic cost
    """
    state_cost = np.sum(X * (X @ Q.T), axis=1)
    control_cost = np.sum(U * (U @ R.T), axis=1)
    return float(np.sum((state_cost + control_cost) * dt))
