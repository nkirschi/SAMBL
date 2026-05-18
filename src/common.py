import dataclasses
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SystemConfig:
    x_dim: int
    u_dim: int
    T: float
    dt: float

    @property
    def H(self):
        """Steps per episode."""
        return int(round(self.T / self.dt))


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Full configuration for one experimental benchmark.

    Coefficient distribution for nonzeros in block X:
        magnitude ~ Uniform(coeff_lower, 2 * x_scale - coeff_lower)
        sign      ~ Uniform({-1, +1})
    so that E[|coeff|] = x_scale and the minimum magnitude = coeff_lower.
    Constraint: x_scale > coeff_lower.
    """

    # ── Dimensions ──────────────────────────────────────────────────
    x_dim: int
    u_dim: int

    # ── Sparsity (per-row, separate A and B blocks) ──────────────────
    s_A: int  # nonzeros per row in the A block
    s_B: int  # nonzeros per row in the B block

    # ── Coefficient distribution ─────────────────────────────────────
    a_scale: float = 0.5  # expected |A_ij| for nonzero entries
    b_scale: float = 0.5  # expected |B_ij| for nonzero entries
    coeff_lower: float = 0.1  # minimum nonzero magnitude (signal gap)

    # ── System ───────────────────────────────────────────────────────
    max_instability: float = 1.0

    # ── Simulation ───────────────────────────────────────────────────
    T: float = 1.0
    dt: float = 0.025
    sigma: float = 0.5
    x0_std: float = 0.0  # 0.0 = start at origin

    # ── Cost matrices ────────────────────────────────────────────────
    q_diag: float = 1.0  # Q = q_diag * I_d
    r_diag: float = 1.0  # R = r_diag * I_p

    # ── Episodes and seeds ───────────────────────────────────────────
    max_episodes: int = 100
    n_seeds: int = 50

    # ── Agents to run ────────────────────────────────────────────────
    agents: tuple = (
        "oracle",
        "dense_greedy",
        "dense_excited",
        "sparse_greedy",
        "sparse_excited",
    )

    # ── Estimation ───────────────────────────────────────────────────
    lambda_lasso: float = None
    c_lambda: float = 2.0
    delta: float = 0.05
    mu_ridge: float = 0.01
    lasso_max_iter: int = 5000
    lasso_tol: float = 1e-4

    # ── Excitation ───────────────────────────────────────────────────
    sigma_u: float = 0.1

    # ── Diagnostics ──────────────────────────────────────────────────
    basin_thresholds: tuple = (0.05, 0.10, 0.15, 0.20, 0.30)
    support_threshold: float = 0.05

    def __post_init__(self):
        if isinstance(self.agents, list):
            object.__setattr__(self, "agents", tuple(self.agents))
        if isinstance(self.basin_thresholds, list):
            object.__setattr__(self, "basin_thresholds", tuple(self.basin_thresholds))

    @property
    def H(self):
        return int(round(self.T / self.dt))

    @property
    def sparsity(self):
        """Total nonzeros per row = s_A + s_B."""
        return self.s_A + self.s_B

    @property
    def sigma_bar(self):
        return self.sigma / np.sqrt(self.dt)

    @property
    def system_config(self):
        return SystemConfig(x_dim=self.x_dim, u_dim=self.u_dim, T=self.T, dt=self.dt)

    @property
    def theoretical_speedup(self):
        dp = self.x_dim + self.u_dim
        return dp / (self.sparsity * np.log(dp))

    @property
    def theoretical_lambda_schedule(self):
        H = self.H

        def get_lambda(N):
            d, p, M = self.x_dim, self.u_dim, self.max_episodes
            log_term = np.log((d + p) * M * d / self.delta)
            return self.c_lambda * self.sigma_bar * np.sqrt(log_term / N)

        return [float(get_lambda(m * H)) for m in range(1, self.max_episodes + 1)]

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML benchmark file."""
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "ExperimentConfig":
        sys_ = raw.get("system", {})
        sim = raw.get("simulation", {})
        cost = raw.get("cost", {})
        train = raw.get("training", {})
        est = raw.get("estimation", {})
        exc = raw.get("excitation", {})
        agents = raw.get("agents", None)
        return cls(
            x_dim=sys_["x_dim"],
            u_dim=sys_["u_dim"],
            s_A=sys_["s_A"],
            s_B=sys_["s_B"],
            a_scale=sys_.get("a_scale", 0.5),
            b_scale=sys_.get("b_scale", 0.5),
            coeff_lower=sys_.get("coeff_lower", 0.1),
            max_instability=sys_.get("max_instability", 1.0),
            T=sim.get("T", 1.0),
            dt=sim.get("dt", 0.025),
            sigma=sim.get("sigma", 0.5),
            x0_std=sim.get("x0_std", 0.0),
            q_diag=cost.get("q_diag", 1.0),
            r_diag=cost.get("r_diag", 1.0),
            max_episodes=train.get("max_episodes", 100),
            n_seeds=train.get("n_seeds", 50),
            agents=tuple(agents)
            if agents is not None
            else (
                "oracle",
                "dense_greedy",
                "dense_excited",
                "sparse_greedy",
                "sparse_excited",
            ),
            lambda_lasso=est.get("lambda_lasso", None),
            c_lambda=est.get("c_lambda", 2.0),
            delta=est.get("delta", 0.05),
            mu_ridge=est.get("mu_ridge", 0.01),
            lasso_max_iter=est.get("lasso_max_iter", 5000),
            lasso_tol=float(est.get("lasso_tol", 1e-4)),
            sigma_u=exc.get("sigma_u", 0.1),
        )

    @classmethod
    def apply_overrides(
        cls, base: "ExperimentConfig", overrides: dict
    ) -> "ExperimentConfig":
        """Return a new ExperimentConfig with override fields applied."""
        kwargs = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        kwargs.update(overrides)
        return cls(**kwargs)
