import numpy as np
import pytest

from common import SystemConfig, EstimatorConfig
from estimator import (
    RegressionBuffer,
    DiscreteRidgeEstimator,
    RowLassoEstimator,
)

@pytest.fixture
def default_dims():
    return {
        "x_dim": 4,
        "u_dim": 2,
        "max_episodes": 5,
        "steps_per_episode": 20,
    }

@pytest.fixture
def configs(default_dims):
    sys_cfg = SystemConfig(
        x_dim=default_dims["x_dim"], u_dim=default_dims["u_dim"], 
        s_A=2, s_B=1, a_scale=0.5, b_scale=0.5, coeff_lower=0.1, 
        max_instability=1.0, sigma=0.5, dt=0.01, T=1.0
    )
    est_cfg = EstimatorConfig(
        mu_ridge=1e-10, lambda_lasso=None, c_lambda=2.0, delta=0.05, 
        lasso_max_iter=1000, lasso_tol=1e-6
    )
    return sys_cfg, est_cfg

@pytest.fixture
def synthetic_data(default_dims):
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    H = default_dims["steps_per_episode"]
    
    Theta_true = np.zeros((d, d + p))
    Theta_true[0, 0] = 0.8        
    Theta_true[1, d] = -0.5       
    Theta_true[3, 2] = 0.9        
    
    rng = np.random.default_rng(42)
    zs = rng.normal(0, 1, size=(H, d + p))
    ys = zs @ Theta_true.T
    
    return zs, ys, Theta_true

@pytest.mark.quick
def test_ridge_estimator_perfect_recovery(default_dims, configs, synthetic_data):
    _, est_cfg = configs
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    H = default_dims["steps_per_episode"]
    zs, ys, Theta_true = synthetic_data
    
    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)
    
    est = DiscreteRidgeEstimator(est_cfg)
    Theta_hat = est.fit(buf.Z, buf.Y)
    
    np.testing.assert_allclose(Theta_hat, Theta_true, atol=1e-6)


@pytest.mark.quick
def test_lasso_estimator_sparse_recovery(default_dims, configs, synthetic_data):
    sys_cfg, _ = configs
    est_cfg = EstimatorConfig(
        mu_ridge=1e-10, lambda_lasso=0.01, c_lambda=2.0, delta=0.05, 
        lasso_max_iter=1000, lasso_tol=1e-6
    )
    
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    H = default_dims["steps_per_episode"]
    zs, ys, Theta_true = synthetic_data
    
    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)
    
    est = RowLassoEstimator(sys_cfg, est_cfg)
    Theta_hat = est.fit(buf.Z, buf.Y)
    
    assert np.abs(Theta_hat[0, 0] - Theta_true[0, 0]) < 0.1
    assert np.abs(Theta_hat[1, d] - Theta_true[1, d]) < 0.1
    
    assert Theta_hat[0, 1] == 0.0
    assert Theta_hat[2, 0] == 0.0

@pytest.mark.quick
def test_lasso_warm_starting(default_dims, configs, synthetic_data):
    sys_cfg, _ = configs
    est_cfg = EstimatorConfig(
        mu_ridge=1e-10, lambda_lasso=0.01, c_lambda=2.0, delta=0.05, 
        lasso_max_iter=1000, lasso_tol=1e-6
    )
    
    d, p = default_dims["x_dim"], default_dims["u_dim"]
    H = default_dims["steps_per_episode"]
    zs, ys, _ = synthetic_data
    
    buf = RegressionBuffer(d, p, max_episodes=2, steps_per_episode=H)
    buf.add_episode(zs, ys)
    
    est = RowLassoEstimator(sys_cfg, est_cfg)
    
    assert not hasattr(est.models[0], "coef_")
    Theta_hat = est.fit(buf.Z, buf.Y)
    assert hasattr(est.models[0], "coef_")
    np.testing.assert_allclose(est.models[0].coef_, Theta_hat[0, :])