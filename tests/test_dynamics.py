import numpy as np
import pytest

from dynamics import ContinuousLQREnv

@pytest.fixture
def config_dt():
    return 0.01

@pytest.fixture
def config_T():
    return 10.0

def test_deterministic_dynamics(config_dt, config_T):
    """
    Verify that with sigma=0, the system evolves exactly as a deterministic Euler step.
    dx = (Ax + Bu) * dt
    Scenario: Resonance. The amplitude should match 0.5 * t * sin(t).
    Dynamics: x'' = -x + u with u = cos(t)
    """
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    sigma = 0.0
    dt = 0.001
    T = 6.0

    env = ContinuousLQREnv(A, B, sigma, dt)

    times = np.linspace(0.0, T, int(T / dt))
    expected_positions = 0.5 * times * np.sin(times)
    actual_positions = np.empty_like(expected_positions)

    x = np.array([0.0, 0.0])
    for i, t in enumerate(times):
        u = np.array([np.cos(t)])
        noise = np.zeros(2)
        x = env.step(x, u, noise)
        actual_positions[i] = x[0]

    np.testing.assert_allclose(
        actual_positions,
        expected_positions,
        atol=1e-2,
        err_msg="Simulation trajectory diverged from analytical resonance solution",
    )

def test_stochastic_scaling(config_dt):
    """
    Verify that noise scales correctly with sqrt(dt).
    If A=0, B=0, sigma=1.0, then Var(dx) should be approx dt * I.
    """
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    sigma = 1.0
    x0 = np.zeros(2)

    N_samples = 5000
    env = ContinuousLQREnv(A, B, sigma, config_dt)

    x = x0.copy()
    dx_samples = []
    for _ in range(N_samples):
        noise = np.random.standard_normal(2)
        x_next = env.step(x, np.array([0.0]), noise)
        dx_samples.append(x_next - x)
        x = x_next

    dx_samples = np.array(dx_samples)

    sample_var = np.var(dx_samples, axis=0)
    expected_var = config_dt
    np.testing.assert_allclose(sample_var, expected_var, rtol=0.05)

    sample_mean = np.mean(dx_samples, axis=0)
    expected_mean = 0.0
    np.testing.assert_allclose(sample_mean, expected_mean, atol=0.005)