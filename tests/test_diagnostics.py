import numpy as np
import pytest

from diagnostics import (
    relative_parameter_error,
    support_metrics,
    restricted_gram_min_eigenvalue,
    closed_loop_spectral_abscissa,
    episode_cost,
)

class TestRelativeParameterError:
    def test_perfect_estmate_gives_zero(self):
        A_true = np.array([[1.0, 0.5], [0.0, -1.0]])
        B_true = np.array([[0.5], [1.0]])
        Theta_true = np.hstack([A_true, B_true])
        
        err = relative_parameter_error(Theta_true, Theta_true)
        assert err["joint"] == pytest.approx(0.0, abs=1e-15)
        assert err["A"] == pytest.approx(0.0, abs=1e-15)
        assert err["B"] == pytest.approx(0.0, abs=1e-15)

    def test_known_relative_error(self):
        A_true = np.eye(3)
        B_true = np.array([[1.0], [0.0], [0.0]])
        Theta_true = np.hstack([A_true, B_true])
        Theta_est = 2 * Theta_true
        
        err = relative_parameter_error(Theta_est, Theta_true)
        assert err["joint"] == pytest.approx(1.0)
        assert err["A"] == pytest.approx(1.0)
        assert err["B"] == pytest.approx(1.0)

    def test_blocks_are_separated(self):
        A_true = np.eye(2)
        B_true = np.eye(2)
        Theta_true = np.hstack([A_true, B_true])
        
        A_est = A_true + np.ones((2, 2))
        B_est = B_true.copy()
        Theta_est = np.hstack([A_est, B_est])
        
        err = relative_parameter_error(Theta_est, Theta_true)
        assert err["A"] > 0
        assert err["B"] == pytest.approx(0.0, abs=1e-15)

    def test_returns_dict_with_correct_keys(self):
        A = np.eye(2)
        B = np.eye(2)
        Theta = np.hstack([A, B])
        err = relative_parameter_error(Theta, Theta)
        assert set(err.keys()) == {"joint", "A", "B"}

class TestSupportMetrics:
    def test_perfect_recovery(self):
        A_est = np.array([[1.0, 0.0, 0.5], [0.0, 0.7, 0.0]])
        B_est = np.array([[0.0], [0.6]])
        Theta_est = np.hstack([A_est, B_est])
        true_supports = [{0, 2}, {1, 3}]
        
        m = support_metrics(Theta_est, true_supports, threshold=0.1)
        assert m["joint"]["precision"] == pytest.approx(1.0)
        assert m["joint"]["recall"] == pytest.approx(1.0)
        assert m["joint"]["f1"] == pytest.approx(1.0)

    def test_threshold_excludes_small_values(self):
        A_est = np.array([[1.0, 0.04]])
        B_est = np.array([[0.0]])
        Theta_est = np.hstack([A_est, B_est])
        true_supports = [{0}]
        
        m = support_metrics(Theta_est, true_supports, threshold=0.05)
        assert m["joint"]["precision"] == pytest.approx(1.0)
        assert m["joint"]["recall"] == pytest.approx(1.0)

    def test_b_block_index_offset(self):
        A_est = np.array([[1.0, 0.0], [0.0, 1.0]])
        B_est = np.array([[1.0, 0.0], [0.0, 1.0]])
        Theta_est = np.hstack([A_est, B_est])
        true_supports = [{0, 2}, {1, 3}]
        
        m = support_metrics(Theta_est, true_supports, threshold=0.5)
        assert m["A"]["f1"] == pytest.approx(1.0)
        assert m["B"]["f1"] == pytest.approx(1.0)

class TestRestrictedGramMinEigenvalue:
    def test_orthonormal_columns_give_one(self):
        N = 100
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((N, 3))
        gram = Z.T @ Z / N
        L = np.linalg.cholesky(gram)
        Z = Z @ np.linalg.inv(L.T)
        
        true_supports = [{0, 1}, {1, 2}]
        min_eig = restricted_gram_min_eigenvalue(Z, true_supports)
        assert min_eig == pytest.approx(1.0, abs=1e-10)

    def test_singular_columns_give_near_zero(self):
        N = 100
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((N, 3))
        Z[:, 1] = Z[:, 0]
        
        true_supports = [{0, 1}, {2}]
        min_eig = restricted_gram_min_eigenvalue(Z, true_supports)
        assert min_eig == pytest.approx(0.0)

class TestClosedLoopSpectralAbscissa:
    def test_zero_gain_recovers_open_loop(self):
        A_true = np.diag([-1.0, -2.0])
        B_true = np.eye(2)
        K = np.zeros((2, 2))
        
        result = closed_loop_spectral_abscissa(A_true, B_true, K)
        assert result == pytest.approx(-1.0)

    def test_stabilising_gain_makes_abscissa_negative(self):
        A_true = np.eye(2)
        B_true = np.eye(2)
        K = -2.0 * np.eye(2)
        
        result = closed_loop_spectral_abscissa(A_true, B_true, K)
        assert result == pytest.approx(-1.0)

class TestEpisodeCost:
    def test_identity_cost_matches_squared_norms(self):
        xs = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        us = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        Q, R = np.eye(2), np.eye(2)
        dt = 0.5

        result = episode_cost(xs, us, Q, R, dt)
        assert result == pytest.approx(3.0)

    def test_dt_scaling(self):
        xs = np.ones((2, 1))
        us = np.ones((2, 1))
        Q, R = np.eye(1), np.eye(1)
        c1 = episode_cost(xs, us, Q, R, dt=1.0)
        c2 = episode_cost(xs, us, Q, R, dt=2.0)
        assert c2 == pytest.approx(2.0 * c1)