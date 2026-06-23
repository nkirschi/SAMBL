"""
Generates random sparse linear-quadratic systems for benchmarking.

Each row of [A_star | B_star] has exactly s_A nonzeros
in the A block and s_B nonzeros in the B block.

Nonzero magnitudes are drawn uniformly from [a_min, a_max] for A entries
and [b_min, b_max] for B entries, with a random sign applied independently.
The lower bounds define the signal gap required for Lasso support recovery.

Rejection sampling on two criteria:
    (i)  No all-zero columns in B.
    (ii) Stabilisability via the Hautus lemma.
"""

from __future__ import annotations

import numpy as np
from typing import Set, Tuple, List
from numpy.typing import NDArray


def sample_sparse_system(
    d: int,
    p: int,
    s_A: int,
    s_B: int,
    seed: int,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    max_attempts: int = 100,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[Set[int]], int]:
    """
    Sample a sparse, stabilisable LQ system.
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

        for i in range(d):
            a_cols = rng.choice(d, size=s_A, replace=False)
            b_cols = rng.choice(p, size=s_B, replace=False)

            for j in a_cols:
                A[i, j] = rng.choice([-1, 1]) * rng.uniform(a_min, a_max)
            for j in b_cols:
                B[i, j] = rng.choice([-1, 1]) * rng.uniform(b_min, b_max)

            supports.append({int(j) for j in a_cols} | {int(j + d) for j in b_cols})

        # (i) No inactive control direction
        if np.any(np.all(B == 0, axis=0)):
            continue

        # (ii) Stabilisability
        if not _is_stabilisable(A, B):
            continue

        return A, B, supports, attempt

    raise RuntimeError(
        f"Failed to sample a valid system after {max_attempts} attempts. "
        f"Consider increasing max_attempts."
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