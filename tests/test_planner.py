import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from common import SystemConfig
from planner import RiccatiODESolver
from system_generator import sample_synthetic_system

@pytest.fixture
def config():
    return SystemConfig(
        d=1, p=1, s_A=1, s_B=1, a_min=0.1, a_max=1.9, b_min=0.1, b_max=1.9, sigma=1.0, dt=0.01, T=5.0
    )

def test_case_1_tanh(config):
    """
    Case 1: Standard LQR (Zero Terminal Cost)
    Exact: P(t) = tanh(T - t)
    """
    solver = RiccatiODESolver(config, Q=np.array([[1.0]]), R=np.array([[1.0]]))
    solver.solve(A=np.array([[0.0]]), B=np.array([[1.0]]))

    times = np.linspace(0, config.T, 5)
    for t in times:
        K = solver.get_K(t, B=np.array([[1.0]]))
        P_numeric = -K[0, 0]
        P_exact = np.tanh(config.T - t)

        assert P_numeric == pytest.approx(P_exact, abs=1e-6)

def test_case_2_rational(config):
    """
    Case 2: Energy Saver (High Terminal Cost)
    Exact: P(t) = G / (1 + G(T-t))
    """
    G = 10.0
    solver = RiccatiODESolver(config, Q=np.array([[0.0]]), R=np.array([[1.0]]))

    solver.solve(
        A=np.array([[0.0]]), B=np.array([[1.0]]), terminal_cost=np.array([[G]])
    )

    times = np.linspace(0, config.T, 5)
    for t in times:
        K = solver.get_K(t, B=np.array([[1.0]]))
        P_numeric = -K[0, 0]
        P_exact = G / (1 + G * (config.T - t))

        assert P_numeric == pytest.approx(P_exact, abs=1e-6)

def test_case_3_damped(config):
    """
    Case 3: Damped Regulator
    Exact: P(t) = 2 / (3 * exp(2(T-t)) - 1)
    """
    solver = RiccatiODESolver(config, Q=np.array([[0.0]]), R=np.array([[1.0]]))

    solver.solve(
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        terminal_cost=np.array([[1.0]]),
    )

    times = np.linspace(0, config.T, 5)
    for t in times:
        K = solver.get_K(t, B=np.array([[1.0]]))
        P_numeric = -K[0, 0]

        tau = config.T - t
        P_exact = 2.0 / (3.0 * np.exp(2 * tau) - 1.0)

        assert P_numeric == pytest.approx(P_exact, abs=1e-6)

# ---------------------------------------------------------------------------
# Solver accuracy on representative sparse systems.
#
# The production solver uses the exact Hamiltonian matrix-exponential recurrence
# (_integrate_expm), with explicit DOP853 and implicit Radau as fallbacks. These
# tests pin its accuracy against an independent tight-tolerance integration and
# against the algebraic Riccati (CARE) limit, and check that the Hamiltonian,
# DOP853, and Radau paths all agree.
# ---------------------------------------------------------------------------


def _representative(d, p, seed):
    A, B, _, _ = sample_synthetic_system(
        d=d, p=p, s_A=min(4, d), s_B=min(2, p), seed=seed,
        a_min=0.1, a_max=1.9, b_min=0.1, b_max=1.9,
    )
    return A, B, np.eye(d), np.eye(p)


def _config(d, p, dt=0.025, T=1.0):
    return SystemConfig(
        d=d, p=p, s_A=min(4, d), s_B=min(2, p),
        a_min=0.1, a_max=1.9, b_min=0.1, b_max=1.9, sigma=1.0, dt=dt, T=T,
    )


def _reference_grid(A, B, Q, R, T, dt, H):
    """Independent ground truth: tight Radau integration of the RDE."""
    d = A.shape[0]
    S = B @ np.linalg.solve(R, B.T)

    def rhs(_, p_flat):
        P = p_flat.reshape(d, d)
        return (A.T @ P + P @ A - P @ S @ P + Q).ravel()

    sol = solve_ivp(
        rhs, [0, T], np.zeros(d * d), method="Radau",
        rtol=1e-12, atol=1e-14, dense_output=True,
    )
    assert sol.success
    return np.stack([sol.sol(j * dt).reshape(d, d) for j in range(H + 1)])


def _rel_fro(P, P_ref):
    num = np.linalg.norm(P - P_ref, axis=(1, 2))
    den = np.maximum(np.linalg.norm(P_ref, axis=(1, 2)), 1e-300)
    return float(np.max(num / den))


@pytest.mark.parametrize("d,p", [(5, 2), (10, 3), (20, 5)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solver_matches_tight_reference(d, p, seed):
    """Production (DOP853) P-grid matches a tight independent integrator."""
    cfg = _config(d, p)
    A, B, Q, R = _representative(d, p, seed)

    solver = RiccatiODESolver(cfg, Q, R)
    assert solver.solve(A, B)

    ref = _reference_grid(A, B, Q, R, cfg.T, cfg.dt, cfg.H)
    assert _rel_fro(solver._P_grid, ref) < 1e-6


@pytest.mark.parametrize("d,p", [(5, 2), (10, 3), (20, 5)])
def test_solution_symmetric_and_psd(d, p):
    cfg = _config(d, p)
    A, B, Q, R = _representative(d, p, seed=0)
    solver = RiccatiODESolver(cfg, Q, R)
    assert solver.solve(A, B)

    for P in solver._P_grid:
        assert np.linalg.norm(P - P.T) < 1e-10
        assert np.linalg.eigvalsh(P)[0] > -1e-9


@pytest.mark.parametrize("d,p", [(6, 2), (10, 3)])
def test_care_limit(d, p):
    """At a long horizon, P(0) converges to the algebraic Riccati solution."""
    cfg = _config(d, p, dt=0.05, T=60.0)
    A, B, Q, R = _representative(d, p, seed=0)
    solver = RiccatiODESolver(cfg, Q, R)
    assert solver.solve(A, B)

    P0 = solver._P_grid[cfg.H]  # tau = T  (physics time t = 0)
    P_care = solve_continuous_are(A, B, Q, R)
    assert np.linalg.norm(P0 - P_care) / np.linalg.norm(P_care) < 1e-5


@pytest.mark.parametrize("d,p", [(5, 2), (10, 3), (20, 5)])
@pytest.mark.parametrize("shift", [0.0, 5.0])
def test_hamiltonian_matches_tight_reference(d, p, shift):
    """The exact Hamiltonian recurrence (_integrate_expm) matches a tight, independent
    integrator -- on stable (shift=0) and near-unstable (shift=5) systems alike, where
    the matrix exponential is exact and stiffness-free. This forces the Hamiltonian path
    directly (rather than going through solve(), which could fall back to an integrator)."""
    cfg = _config(d, p)
    A, B, Q, R = _representative(d, p, seed=0)
    A = A + shift * np.eye(d)  # push toward instability (stiff for explicit integrators)
    solver = RiccatiODESolver(cfg, Q, R)
    S = B @ np.linalg.solve(R, B.T)
    ham = solver._integrate_expm(A, S, None)
    assert ham is not None
    ref = _reference_grid(A, B, Q, R, cfg.T, cfg.dt, cfg.H)
    assert _rel_fro(ham, ref) < 1e-6


def test_dop853_matches_radau_fallback():
    """The primary (DOP853) and stiff-fallback (Radau) paths agree."""
    cfg = _config(10, 3)
    A, B, Q, R = _representative(10, 3, seed=0)
    solver = RiccatiODESolver(cfg, Q, R)
    S = B @ np.linalg.solve(R, B.T)

    dop = solver._integrate(A, S, None, "DOP853")
    radau = solver._integrate(A, S, None, "Radau")
    assert dop is not None and radau is not None
    assert _rel_fro(dop, radau) < 1e-6


def test_get_K_interpolates_off_grid():
    """get_K is exact on grid points and continuous between them."""
    cfg = _config(5, 2)
    A, B, Q, R = _representative(5, 2, seed=0)
    solver = RiccatiODESolver(cfg, Q, R)
    assert solver.solve(A, B)

    # midpoint of the first interval should sit between the two grid gains
    k0 = solver.get_K(0.0, B)
    k_half = solver.get_K(0.5 * cfg.dt, B)
    k1 = solver.get_K(cfg.dt, B)
    assert np.all((k_half <= np.maximum(k0, k1) + 1e-12))
    assert np.all((k_half >= np.minimum(k0, k1) - 1e-12))
