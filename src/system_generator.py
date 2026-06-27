"""
Samplers that build the row-sparse linear-quadratic systems we benchmark on.
All return (A, B, supports, attempt) with d-dimensional state and p-dimensional control.
"""

from __future__ import annotations

import numpy as np
from typing import Set, Tuple, List
from numpy.typing import NDArray

from ieee39_data import (
    IEEE39_BRANCHES,
    IEEE39_GEN_H,
    IEEE39_ACTUATED_BUSES,
    IEEE39_N_BUS,
)


def sample_synthetic_system(
    d: int,
    p: int,
    s_A: int,
    s_B: int,
    seed: int,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    normalise_B: bool = False,
    max_attempts: int = 100,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[Set[int]], int]:
    """
    Generates random sparse linear-quadratic systems for benchmarking.

    Each row of [A_star | B_star] has exactly s_A nonzeros
    in the A block and s_B nonzeros in the B block.

    Nonzero magnitudes are drawn uniformly from [a_min, a_max] for A entries
    and [b_min, b_max] for B entries, with a random sign applied independently.
    The lower bounds define the signal gap required for Lasso support recovery.

    Rejection sampling based on stabilisability via the Hautus lemma.
    """
    assert 0 < s_A <= d
    assert 0 < s_B <= p
    assert 0 < a_min < a_max
    assert 0 < b_min < b_max

    rng = np.random.default_rng(seed)

    for attempt in range(1, max_attempts + 1):
        A = np.zeros((d, d), dtype=np.float64)
        B = np.zeros((d, p), dtype=np.float64)
        supports = []
        b_cols_per_row = _assign_covering_columns(rng, d, p, s_B)

        for i in range(d):
            a_cols = rng.choice(d, size=s_A, replace=False)
            b_cols = b_cols_per_row[i]

            for j in a_cols:
                A[i, j] = rng.choice([-1, 1]) * rng.uniform(a_min, a_max)
            for j in b_cols:
                B[i, j] = rng.choice([-1, 1]) * rng.uniform(b_min, b_max)

            supports.append({int(j) for j in a_cols} | {int(j + d) for j in b_cols})

        # optional B column normalisation
        if normalise_B:
            B = B / np.linalg.norm(B, axis=0)

        # stabilisability check
        if not _is_stabilisable(A, B):
            continue

        return A, B, supports, attempt

    raise RuntimeError(
        f"Failed to sample a synthetic system after {max_attempts} attempts. "
        f"Consider increasing max_attempts."
    )


def _assign_covering_columns(rng, d: int, p: int, s_B: int) -> List[NDArray[np.int_]]:
    """
    Assign s_B distinct control columns to each of the d rows such that every one
    of the p columns is hit by at least one row. Each of the p columns is first
    placed on a distinct random row (requires p <= d); rows are then filled up to
    s_B with uniformly random distinct columns.
    """
    assert p <= d, "need p <= d so each control column can be covered by a row"
    rows = [set() for _ in range(d)]
    cover_rows = rng.permutation(d)[:p]  # p distinct rows, one per column
    for col, row in enumerate(cover_rows):
        rows[int(row)].add(col)
    for i in range(d):
        while len(rows[i]) < s_B:
            rows[i].add(int(rng.integers(p)))
    return [np.array(sorted(s), dtype=int) for s in rows]


def sample_spring_chain(
    d: int,
    p: int,
    seed: int,
    k_min: float = 0.5,
    k_max: float = 2.0,
    m_min: float = 0.5,
    m_max: float = 1.5,
    max_attempts: int = 100,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[Set[int]], int]:
    """
    Randomized undamped spring-mass chain in interleaved state-space form
    x = [q_1, qdot_1, ..., q_n, qdot_n], d = 2n, fixed walls at both ends.

    Nearest-neighbour stiffness coupling gives a banded A with row-sparsity fixed
    independent of n: position rows have s=1 (dq/dt = qdot), interior velocity rows
    s=3 (undamped coupling to q_{i-1}, q_i, q_{i+1}).
    The p actuated masses are evenly spaced, each a single 1/m entry in B.

    Per seed the masses m_i (in [m_min, m_max]) and spring constants k_j (in
    [k_min, k_max], including the two wall springs) are drawn uniformly -- fixed
    structure, random values -- which keeps the design heterogeneous (irregular modes).

    Rejection sampling based on stabilisability via the Hautus lemma.
    """
    assert d % 2 == 0, "spring chain requires d = 2n (even state dimension)"
    n = d // 2
    assert 0 < p <= n, "number of actuators p must satisfy 0 < p <= n = d/2"

    act_masses = (np.arange(p) * n) // p  # p distinct, evenly-spaced mass indices

    rng = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        m = rng.uniform(m_min, m_max, size=n)  # masses
        k = rng.uniform(k_min, k_max, size=n + 1)  # springs (+ 2 walls)
        A = np.zeros((d, d), dtype=np.float64)
        B = np.zeros((d, p), dtype=np.float64)
        supports: List[Set[int]] = [set() for _ in range(d)]

        for i in range(n):
            pos, vel = 2 * i, 2 * i + 1
            A[pos, vel] = 1.0  # dq_i/dt = qdot_i
            supports[pos].add(vel)
            kl, kr = k[i], k[i + 1]  # left / right spring of mass i
            A[vel, pos] = -(kl + kr) / m[i]
            supports[vel].add(pos)
            if i > 0:
                A[vel, 2 * (i - 1)] = kl / m[i]
                supports[vel].add(2 * (i - 1))
            if i < n - 1:
                A[vel, 2 * (i + 1)] = kr / m[i]
                supports[vel].add(2 * (i + 1))

        for j, a in enumerate(act_masses):
            B[2 * int(a) + 1, j] = 1.0 / m[a]
            supports[2 * int(a) + 1].add(d + j)

        if not _is_stabilisable(A, B):
            continue

        return A, B, supports, attempt

    raise RuntimeError(
        f"Failed to sample a stabilisable spring chain after {max_attempts} attempts."
        f"Consider increasing max_attempts."
    )


def sample_ieee39(
    seed: int,
    m_load: float = 10.0,
    damping: float = 3.0,
    jitter: float = 0.0,
    max_attempts: int = 50,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[Set[int]], int]:
    """
    Build the IEEE 39-bus (New England) grid as a swing-oscillator network in interleaved
    state-space form x = [theta_1, omega_1, ..., theta_n, omega_n], d = 2n = 78. The
    susceptance-weighted Laplacian (L_ij = -1/x_ij) drives the frequency rows, so A is
    row-sparse on the irregular grid graph (topology/inertia data in ``ieee39_data``).

    The nine generator buses are actuated (p = 9). The heavy reference machine (bus 39),
    is unactuated. Generators use the real inertias, load buses ``m_load``, with uniform
    ``damping``. Control columns are scaled by the median generator inertia (B_i = M_ref / M_i)
    to lift control authority above the noise floor. With ``jitter > 0`` the
    susceptances and inertias are perturbed log-normally per seed for a heterogeneous
    ensemble, otherwise only the noise varies across seeds.

    Rejection sampling based on stabilisability via the Hautus lemma.
    """
    n = IEEE39_N_BUS
    d = 2 * n
    p = len(IEEE39_ACTUATED_BUSES)
    m_ref = float(np.median([IEEE39_GEN_H[b] for b in IEEE39_ACTUATED_BUSES]))

    rng = np.random.default_rng(seed)
    for attempt in range(1, max_attempts + 1):
        L = np.zeros((n, n))
        for f, t, x in IEEE39_BRANCHES:
            b = 1.0 / abs(x)
            if jitter:
                b *= float(np.exp(rng.normal(0.0, jitter)))
            i, j = f - 1, t - 1
            L[i, j] -= b
            L[j, i] -= b
            L[i, i] += b
            L[j, j] += b
        M = np.array([IEEE39_GEN_H.get(bus, m_load) for bus in range(1, n + 1)])
        if jitter:
            M = M * np.exp(rng.normal(0.0, jitter, size=n))

        A = np.zeros((d, d), dtype=np.float64)
        B = np.zeros((d, p), dtype=np.float64)
        supports: List[Set[int]] = [set() for _ in range(d)]
        for i in range(n):
            th, om = 2 * i, 2 * i + 1
            A[th, om] = 1.0  # theta_dot = omega
            supports[th].add(om)
            A[om, om] = -damping / M[i]  # damping term
            supports[om].add(om)
            for j in np.nonzero(L[i])[0]:  # susceptance couplings
                A[om, 2 * int(j)] = -L[i, j] / M[i]
                supports[om].add(2 * int(j))
        for col, bus in enumerate(IEEE39_ACTUATED_BUSES):
            i = bus - 1
            B[2 * i + 1, col] = m_ref / M[i]
            supports[2 * i + 1].add(d + col)

        if not _is_stabilisable(A, B):
            continue

        return A, B, supports, attempt

    raise RuntimeError(
        f"IEEE 39-bus system failed stabilisability after {max_attempts} attempts."
    )


def _is_stabilisable(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    rtol: float = 1e-8,
) -> bool:
    """
    Check stabilisability via the Hautus lemma.

    (A, B) is stabilisable iff rank([lambda*I - A, B]) = d
    for every eigenvalue lambda of A with Re(lambda) >= 0.

    References
    ----------
    Hespanha, "Linear Systems Theory" (2018), Theorem 14.3
    """
    d = A.shape[0]
    eigenvalues = np.linalg.eigvals(A)
    for lam in eigenvalues:
        if np.real(lam) >= 0:
            block = np.hstack([lam * np.eye(d) - A, B])
            if np.linalg.matrix_rank(block, rtol=rtol) < d:
                return False
    return True


def _is_controllable(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    rtol: float = 1e-8,
) -> bool:
    """
    Check controllability via the Hautus lemma.

    (A, B) is controllable iff rank([lambda*I - A, B]) = d
    for every eigenvalue lambda of A.

    References
    ----------
    Hespanha, "Linear Systems Theory" (2018), Theorem 12.3
    """
    d = A.shape[0]
    eigenvalues = np.linalg.eigvals(A)
    for lam in eigenvalues:
        block = np.hstack([lam * np.eye(d) - A, B])
        if np.linalg.matrix_rank(block, rtol=rtol) < d:
            return False
    return True
