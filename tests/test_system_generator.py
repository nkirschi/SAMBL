"""
Unit tests for system_generator.py.
Run with:  pytest test_system_generator.py -v
"""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov, LinAlgError

from system_generator import (
    sample_sparse_system,
    _is_stabilisable,
    _is_controllable,
    define_cost_matrices,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _sample(d=6, p=2, s_A=2, s_B=1, seed=0, **kw):
    """Convenience wrapper with sensible defaults."""
    return sample_sparse_system(d, p, s_A, s_B, seed, **kw)


def _gramian_controllable(A, B, tol=1e-10):
    try:
        W = solve_continuous_lyapunov(A, -B @ B.T)
        W = (W + W.T) / 2.0
        return float(np.min(np.linalg.eigvalsh(W))) > tol
    except (LinAlgError, Exception):
        return False


# ── _is_stabilisable ──────────────────────────────────────────────────


class TestIsStabilisable:

    def test_stable_system_always_stabilisable(self):
        A = np.diag([-1.0, -2.0, -0.5])
        B = np.zeros((3, 1))
        assert _is_stabilisable(A, B) is True

    def test_unstable_mode_reachable(self):
        A = np.diag([0.5, -1.0])
        B = np.array([[1.0], [0.0]])
        assert _is_stabilisable(A, B) is True

    def test_unstable_mode_unreachable(self):
        A = np.diag([0.5, -1.0])
        B = np.array([[0.0], [1.0]])
        assert _is_stabilisable(A, B) is False

    def test_two_unstable_modes_both_reachable(self):
        A = np.diag([0.5, 1.0, -1.0])
        B = np.eye(3)[:, :2]
        assert _is_stabilisable(A, B) is True

    def test_two_unstable_modes_one_unreachable(self):
        A = np.diag([0.5, 1.0, -1.0])
        B = np.array([[1.0], [0.0], [0.0]])
        assert _is_stabilisable(A, B) is False

    def test_marginal_eigenvalue_is_checked(self):
        A = np.diag([0.0, -1.0])
        assert _is_stabilisable(A, np.array([[1.0], [0.0]])) is True
        assert _is_stabilisable(A, np.array([[0.0], [1.0]])) is False

    def test_stabilisable_not_controllable(self):
        A = np.diag([0.5, -1.0])
        B = np.array([[1.0], [0.0]])
        assert _is_stabilisable(A, B) is True
        assert _is_controllable(A, B) is False

    def test_precomputed_eigenvalues_give_same_result(self):
        A = np.diag([0.5, -1.0])
        B = np.array([[1.0], [0.0]])
        eigs = np.linalg.eigvals(A)
        assert _is_stabilisable(A, B, eigenvalues=eigs) == _is_stabilisable(A, B)


# ── _is_controllable ──────────────────────────────────────────────────


class TestIsControllable:

    def test_companion_form_is_controllable(self):
        A = np.array([[-1.0, 1.0], [0.0, -2.0]])
        B = np.array([[0.0], [1.0]])
        assert _is_controllable(A, B) is True

    def test_decoupled_unreachable_mode_not_controllable(self):
        A = np.diag([-1.0, -2.0])
        B = np.array([[1.0], [0.0]])
        assert _is_controllable(A, B) is False

    def test_identity_B_is_controllable(self):
        A = np.diag([-1.0, -1.5, -2.0])
        B = np.eye(3)
        assert _is_controllable(A, B) is True

    def test_agrees_with_gramian_on_stable_systems(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            d = int(rng.integers(2, 6))
            p = int(rng.integers(1, d + 1))
            A = rng.standard_normal((d, d)) * 0.3
            shift = np.max(np.real(np.linalg.eigvals(A))) + 0.5
            if shift > 0:
                A -= shift * np.eye(d)
            B = rng.standard_normal((d, p)) * 0.5
            assert _is_controllable(A, B) == _gramian_controllable(A, B)

    def test_controllable_implies_stabilisable(self):
        rng = np.random.default_rng(1)
        for _ in range(30):
            d = int(rng.integers(2, 5))
            p = int(rng.integers(1, d + 1))
            A = rng.standard_normal((d, d)) * 0.5
            B = rng.standard_normal((d, p)) * 0.5
            if _is_controllable(A, B):
                assert _is_stabilisable(A, B)


# ── sample_sparse_system ──────────────────────────────────────────────


class TestSampleSparseSystem:

    def test_returns_correct_shapes(self):
        A, B, supports, _ = _sample(d=6, p=2, s_A=2, s_B=1)
        assert A.shape == (6, 6)
        assert B.shape == (6, 2)
        assert len(supports) == 6

    def test_reproducible_with_same_seed(self):
        A1, B1, _, _ = _sample(seed=42)
        A2, B2, _, _ = _sample(seed=42)
        assert np.allclose(A1, A2) and np.allclose(B1, B2)

    def test_different_seeds_give_different_systems(self):
        A1, _, _, _ = _sample(seed=0)
        A2, _, _, _ = _sample(seed=1)
        assert not np.allclose(A1, A2)

    def test_exact_row_sparsity_in_a_block(self):
        d, p, s_A, s_B = 8, 3, 2, 1
        A, B, supports, _ = _sample(d=d, p=p, s_A=s_A, s_B=s_B, seed=7)
        for i, supp in enumerate(supports):
            a_cols = {j for j in supp if j < d}
            b_cols = {j - d for j in supp if j >= d}
            assert len(a_cols) == s_A, f"Row {i}: A has {len(a_cols)} nonzeros, expected {s_A}"
            assert len(b_cols) == s_B, f"Row {i}: B has {len(b_cols)} nonzeros, expected {s_B}"

    def test_coefficient_magnitudes_in_expected_range(self):
        a_scale, b_scale, lower = 0.6, 0.4, 0.1
        A, B, supports, _ = _sample(
            d=8, p=3, s_A=2, s_B=1, seed=5,
            a_scale=a_scale, b_scale=b_scale, coeff_lower=lower,
        )
        a_upper = 2 * a_scale - lower
        b_upper = 2 * b_scale - lower
        for i, supp in enumerate(supports):
            for j in supp:
                if j < 8:
                    assert lower <= abs(A[i, j]) <= a_upper + 1e-12
                else:
                    assert lower <= abs(B[i, j - 8]) <= b_upper + 1e-12

    def test_no_all_zero_b_columns(self):
        A, B, _, _ = _sample(d=8, p=3, s_A=2, s_B=1, seed=3)
        assert not np.any(np.all(B == 0, axis=0))

    def test_max_instability_respected(self):
        for seed in range(20):
            A, _, _, _ = _sample(seed=seed, max_instability=0.8)
            assert float(np.max(np.real(np.linalg.eigvals(A)))) <= 0.8 + 1e-10

    def test_sampled_system_is_stabilisable(self):
        for seed in range(20):
            A, B, _, _ = _sample(seed=seed)
            assert _is_stabilisable(A, B)

    def test_supports_are_valid_index_sets(self):
        d, p = 8, 3
        _, _, supports, _ = _sample(d=d, p=p, s_A=2, s_B=1, seed=0)
        for supp in supports:
            assert all(0 <= j < d + p for j in supp)

    def test_accepts_unstable_systems(self):
        has_unstable = any(
            float(np.max(np.real(np.linalg.eigvals(_sample(seed=s)[0])))) > 0
            for s in range(50)
        )
        assert has_unstable, "No unstable systems accepted: stability shift may still be active"

    def test_looser_instability_bound_accepts_more(self):
        tight = [_sample(seed=s, max_instability=0.3)[3] for s in range(20)]
        loose = [_sample(seed=s, max_instability=2.0)[3] for s in range(20)]
        assert np.mean(loose) <= np.mean(tight)

    def test_raises_on_impossible_config(self):
        with pytest.raises(RuntimeError, match="Failed to sample"):
            sample_sparse_system(
                d=20, p=1, s_A=1, s_B=1, seed=0,
                max_instability=0.01, max_attempts=5,
            )

    def test_scale_affects_coefficient_magnitude(self):
        """Higher scale should produce larger absolute coefficients on average."""
        def mean_abs(scale):
            A, B, supports, _ = _sample(
                d=10, p=3, s_A=2, s_B=1, seed=99,
                a_scale=scale, b_scale=scale,
            )
            vals = []
            for i, supp in enumerate(supports):
                for j in supp:
                    vals.append(abs(A[i, j]) if j < 10 else abs(B[i, j - 10]))
            return float(np.mean(vals))
        assert mean_abs(0.3) < mean_abs(0.7)


# ── define_cost_matrices ──────────────────────────────────────────────


class TestDefineCostMatrices:

    def test_returns_scaled_identities(self):
        Q, R = define_cost_matrices(5, 3, q_diag=2.0, r_diag=0.5)
        assert np.allclose(Q, 2.0 * np.eye(5))
        assert np.allclose(R, 0.5 * np.eye(3))

    def test_default_is_identity(self):
        Q, R = define_cost_matrices(4, 2)
        assert np.allclose(Q, np.eye(4))
        assert np.allclose(R, np.eye(2))

    def test_correct_shapes(self):
        for d, p in [(2, 1), (10, 5), (20, 3)]:
            Q, R = define_cost_matrices(d, p)
            assert Q.shape == (d, d) and R.shape == (p, p)

    def test_positive_definite(self):
        Q, R = define_cost_matrices(4, 2, q_diag=0.5, r_diag=2.0)
        assert np.all(np.linalg.eigvalsh(Q) > 0)
        assert np.all(np.linalg.eigvalsh(R) > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
