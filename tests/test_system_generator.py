import numpy as np
import pytest

from system_generator import (
    sample_synthetic_system,
    sample_spring_chain,
    _is_stabilisable,
    _is_controllable,
)


def _sample(d=6, p=2, s_A=2, s_B=1, seed=0, **kwargs):
    default_kwargs = {
        "a_min": 0.1,
        "a_max": 1.9,
        "b_min": 0.1,
        "b_max": 1.9,
    }
    default_kwargs.update(kwargs)
    return sample_synthetic_system(d, p, s_A, s_B, seed, **default_kwargs)


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


class TestSampleSyntheticSystem:
    def test_returns_correct_shapes(self):
        A, B, supports, attempt = _sample(d=6, p=2, s_A=2, s_B=1)
        assert A.shape == (6, 6)
        assert B.shape == (6, 2)
        assert len(supports) == 6
        assert attempt >= 1

    def test_reproducible_with_same_seed(self):
        A1, B1, _, _ = _sample(seed=42)
        A2, B2, _, _ = _sample(seed=42)
        assert np.allclose(A1, A2) and np.allclose(B1, B2)

    def test_exact_row_sparsity_in_a_block(self):
        d, p, s_A, s_B = 8, 3, 2, 1
        A, B, supports, _ = _sample(d=d, p=p, s_A=s_A, s_B=s_B, seed=7)
        for i, supp in enumerate(supports):
            a_cols = {j for j in supp if j < d}
            b_cols = {j - d for j in supp if j >= d}
            assert len(a_cols) == s_A
            assert len(b_cols) == s_B

    def test_coefficient_magnitudes_in_expected_range(self):
        a_min, a_max = 0.1, 1.9
        b_min, b_max = 0.1, 1.9
        A, B, supports, _ = _sample(
            d=8,
            p=3,
            s_A=2,
            s_B=1,
            seed=5,
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max
        )
        for i, supp in enumerate(supports):
            for j in supp:
                if j < 8:
                    assert a_min <= abs(A[i, j]) <= a_max
                else:
                    assert b_min <= abs(B[i, j - 8]) <= b_max

    def test_no_all_zero_b_columns(self):
        A, B, _, _ = _sample(d=8, p=3, s_A=2, s_B=1, seed=3)
        assert not np.any(np.all(B == 0, axis=0))

    def test_raises_on_impossible_config(self):
        with pytest.raises(RuntimeError, match="Failed to sample"):
            sample_synthetic_system(
                d=20,
                p=1,
                s_A=1,
                s_B=1,
                seed=0,
                a_min=0.1,
                a_max=1.9,
                b_min=0.1,
                b_max=1.9,
                max_attempts=5,
            )


# ---------------------------------------------------------------------------
# Spring-mass chain (banded, undamped, randomized parameters)
# ---------------------------------------------------------------------------


class TestSpringChain:
    def test_structure_and_sparsity(self):
        d, n = 20, 10
        A, B, sup, _ = sample_spring_chain(d=d, p=5, seed=0)
        assert A.shape == (d, d) and B.shape == (d, 5)
        for i in range(n):
            # position row: exactly q̇_i, coefficient 1
            assert (A[2 * i] != 0).sum() == 1 and A[2 * i, 2 * i + 1] == 1.0
            # velocity row: 3 nonzeros interior, 2 at the boundary; no velocity coupling
            assert (A[2 * i + 1] != 0).sum() == (3 if 0 < i < n - 1 else 2)
        # each control column hits exactly one (velocity) row
        assert (np.count_nonzero(B, axis=0) == 1).all()
        assert all(r % 2 == 1 for r in np.flatnonzero(B.any(axis=1)))

    def test_banded_and_marginally_stable(self):
        A, B, _, _ = sample_spring_chain(d=20, p=5, seed=1)
        bandwidth = max(abs(i - j) for i, j in zip(*np.nonzero(A)))
        assert bandwidth <= 3  # interleaved nearest-neighbour band
        # undamped -> eigenvalues on the imaginary axis
        assert np.max(np.abs(np.real(np.linalg.eigvals(A)))) < 1e-9

    def test_controllable_across_dimensions(self):
        for d in (10, 20, 50):
            A, B, _, _ = sample_spring_chain(d=d, p=max(1, (d // 2) // 2), seed=0)
            assert _is_controllable(A, B)

    def test_supports_match_pattern(self):
        d = 20
        A, B, sup, _ = sample_spring_chain(d=d, p=5, seed=2)
        for r in range(d):
            nz = set(np.flatnonzero(A[r]).tolist()) | {
                d + c for c in np.flatnonzero(B[r]).tolist()
            }
            assert nz == sup[r]

    def test_deterministic(self):
        A1, B1, _, _ = sample_spring_chain(d=20, p=5, seed=3)
        A2, B2, _, _ = sample_spring_chain(d=20, p=5, seed=3)
        assert np.array_equal(A1, A2) and np.array_equal(B1, B2)

    def test_odd_d_rejected(self):
        with pytest.raises(AssertionError):
            sample_spring_chain(d=21, p=5, seed=0)

    def test_too_many_actuators_rejected(self):
        with pytest.raises(AssertionError):
            sample_spring_chain(d=20, p=11, seed=0)  # p > n=10
