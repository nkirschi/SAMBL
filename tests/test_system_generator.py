import numpy as np
import pytest

from system_generator import sample_sparse_system, _is_stabilisable


def _sample(d=6, p=2, s_A=2, s_B=1, seed=0, **kwargs):
    default_kwargs = {
        "a_scale": 0.5,
        "b_scale": 0.5,
        "coeff_lower": 0.1,
        "max_instability": 1.0,
    }
    default_kwargs.update(kwargs)
    return sample_sparse_system(d, p, s_A, s_B, seed, **default_kwargs)


class TestIsStabilisable:
    def test_stable_system_always_stabilisable(self):
        A = np.diag([-1.0, -2.0, -0.5])
        B = np.zeros((3, 1))
        assert _is_stabilisable(A, B, np.linalg.eigvals(A)) is True

    def test_unstable_mode_reachable(self):
        A = np.diag([0.5, -1.0])
        B = np.array([[1.0], [0.0]])
        assert _is_stabilisable(A, B, np.linalg.eigvals(A)) is True

    def test_unstable_mode_unreachable(self):
        A = np.diag([0.5, -1.0])
        B = np.array([[0.0], [1.0]])
        assert _is_stabilisable(A, B, np.linalg.eigvals(A)) is False


class TestSampleSparseSystem:
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
        a_scale, b_scale, lower = 0.6, 0.4, 0.1
        A, B, supports, _ = _sample(
            d=8,
            p=3,
            s_A=2,
            s_B=1,
            seed=5,
            a_scale=a_scale,
            b_scale=b_scale,
            coeff_lower=lower,
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

    def test_raises_on_impossible_config(self):
        with pytest.raises(RuntimeError, match="Failed to sample"):
            sample_sparse_system(
                d=20,
                p=1,
                s_A=1,
                s_B=1,
                seed=0,
                a_scale=0.5,
                b_scale=0.5,
                coeff_lower=0.1,
                max_instability=0.01,
                max_attempts=5,
            )
