"""
Core experimental loop with matched-seed design.
Each seed defines one system and one noise stream shared across all agents.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

from dynamics import ContinuousLQREnv
from estimator import RegressionBuffer
from system_generator import sample_sparse_system, define_cost_matrices
from agent import (
    OracleAgent,
    DenseGreedyAgent,
    DenseExcitedAgent,
    SparseGreedyAgent,
    SparseExcitedAgent,
)
from diagnostics import collect_diagnostics, episode_cost

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# All known agent names; actual agents run are determined by exp_config.agents
ALL_AGENT_NAMES = (
    "oracle",
    "dense_greedy",
    "dense_excited",
    "sparse_greedy",
    "sparse_excited",
)

_EXCITATION_TYPES = (DenseExcitedAgent, SparseExcitedAgent)


@dataclass
class EpisodeRecord:
    """Per-episode results for one agent."""

    cost: float = 0.0
    excitation_tax: float = 0.0  # sigma_u^2 * tr(R) * T; nonzero for excitation agents
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SeedResult:
    """All results from one seed, for all agents."""

    seed: int = 0
    A_star: np.ndarray = None
    B_star: np.ndarray = None
    supports: list = None
    episodes: Dict[str, List[EpisodeRecord]] = field(default_factory=dict)

    @property
    def agent_names(self):
        return list(self.episodes.keys())

    def cumulative_regret(self, agent_name, oracle_name="oracle"):
        oracle_costs = np.array([ep.cost for ep in self.episodes[oracle_name]])
        agent_costs = np.array([ep.cost for ep in self.episodes[agent_name]])
        return np.cumsum(agent_costs - oracle_costs)

    def adjusted_cumulative_regret(self, agent_name, oracle_name="oracle"):
        """Cumulative regret with the deterministic excitation tax removed."""
        oracle_costs = np.array([ep.cost for ep in self.episodes[oracle_name]])
        agent_costs = np.array([ep.cost for ep in self.episodes[agent_name]])
        taxes = np.array([ep.excitation_tax for ep in self.episodes[agent_name]])
        return np.cumsum(agent_costs - taxes - oracle_costs)

    def final_cumulative_regret(self, agent_name, oracle_name="oracle"):
        return self.cumulative_regret(agent_name, oracle_name)[-1]

    def diagnostic_trajectory(self, agent_name, key):
        return [ep.diagnostics.get(key, np.nan) for ep in self.episodes[agent_name]]


def _make_stabilising_init(d, p, margin=1.0):
    return -margin * np.eye(d), np.zeros(
        (d, p)
    )  # TODO: eliminate the need for placeholder estimates


def _build_lasso_kwargs(exp_config):
    """Build keyword arguments for SparseGreedyAgent / SparseExcitedAgent."""
    common = dict(max_iter=exp_config.lasso_max_iter, tol=exp_config.lasso_tol)
    if exp_config.lambda_lasso is not None:
        return dict(lambda_fixed=exp_config.lambda_lasso, **common)
    return dict(
        sigma_bar=exp_config.sigma_bar,
        c_lambda=exp_config.c_lambda,
        delta=exp_config.delta,
        max_episodes=exp_config.max_episodes,
        **common,
    )


def _create_agents(exp_config, sys_config, Q, R, A_star, B_star, seed):
    """Instantiate the agents listed in exp_config.agents."""
    d, p = exp_config.x_dim, exp_config.u_dim
    A_init, B_init = _make_stabilising_init(d, p)
    lasso_kw = _build_lasso_kwargs(exp_config)
    requested = set(exp_config.agents)
    agents = {}

    if "oracle" in requested:
        agents["oracle"] = OracleAgent(sys_config, Q, R, A_star, B_star)

    if "dense_greedy" in requested:
        agents["dense_greedy"] = DenseGreedyAgent(
            sys_config, Q, R, A_init, B_init, mu=exp_config.mu_ridge
        )

    if "dense_excited" in requested:
        agents["dense_excited"] = DenseExcitedAgent(
            sys_config,
            Q,
            R,
            A_init,
            B_init,
            sigma_u=exp_config.sigma_u,
            excitation_rng=np.random.RandomState(seed + 1_500_000),
            mu=exp_config.mu_ridge,
        )

    if "sparse_greedy" in requested:
        agents["sparse_greedy"] = SparseGreedyAgent(
            sys_config, Q, R, A_init, B_init, **lasso_kw
        )

    if "sparse_excited" in requested:
        agents["sparse_excited"] = SparseExcitedAgent(
            sys_config,
            Q,
            R,
            A_init,
            B_init,
            sigma_u=exp_config.sigma_u,
            excitation_rng=np.random.RandomState(seed + 1_000_000),
            **lasso_kw,
        )

    # Preserve the order specified in exp_config.agents
    return {name: agents[name] for name in exp_config.agents if name in agents}


def run_paired_experiment(exp_config, seed, verbose=False):
    """Run one paired-seed experiment: all configured agents on the same system and noise."""
    d, p = exp_config.x_dim, exp_config.u_dim
    M, H = exp_config.max_episodes, exp_config.H
    dt = exp_config.dt
    sigma = exp_config.sigma

    # Sample system using per-block sparsity and scale params from config
    A_star, B_star, supports, n_attempts = sample_sparse_system(
        d=d,
        p=p,
        s_A=exp_config.s_A,
        s_B=exp_config.s_B,
        seed=seed,
        a_scale=exp_config.a_scale,
        b_scale=exp_config.b_scale,
        coeff_lower=exp_config.coeff_lower,
        max_instability=exp_config.max_instability,
    )
    Q, R = define_cost_matrices(
        d, p, q_diag=exp_config.q_diag, r_diag=exp_config.r_diag
    )

    if verbose:
        print(f"Sampled system after {n_attempts} attempt(s)")

    sys_config = exp_config.system_config

    # Pre-generate shared noise
    noise_rng = np.random.RandomState(seed + 2_000_000)
    shared_noise = noise_rng.randn(M, H, d)

    # Pre-generate shared exploration controls (same for all agents → paired design)
    m_explore = int(np.ceil(2 * (d + p) / H))
    explore_rng = np.random.RandomState(seed + 3_000_000)
    shared_exploration = explore_rng.randn(m_explore, H, p) * sigma

    # Checkpoint episodes for matrix snapshots
    n_checkpoints = min(8, M)
    checkpoint_set = set(
        np.round(np.linspace(0, M - 1, n_checkpoints)).astype(int).tolist()
    )

    # Initial condition
    if exp_config.x0_std > 0:
        x0_rng = np.random.RandomState(seed + 4_000_000)
        x0 = x0_rng.randn(d) * exp_config.x0_std
    else:
        x0 = np.zeros(d)

    agents = _create_agents(exp_config, sys_config, Q, R, A_star, B_star, seed)
    agent_names = list(agents.keys())
    learning_names = [n for n in agent_names if n != "oracle"]

    buffers = {name: RegressionBuffer(d, p, M, H) for name in learning_names}
    Sigma = sigma * np.eye(d)

    result = SeedResult(
        seed=seed,
        A_star=A_star,
        B_star=B_star,
        supports=supports,
        episodes={name: [] for name in agent_names},
    )

    for m in tqdm(range(M), disable=not verbose):
        for name in agent_names:
            agent = agents[name]
            env = ContinuousLQREnv(A_star, B_star, Sigma, x0, sys_config)

            states, controls, zs_list, ys_list = [], [], [], []
            x = x0.copy()

            for k in range(H):
                t = k * dt

                if name != "oracle" and m < m_explore:
                    u = shared_exploration[m, k]
                else:
                    u = agent.get_control(t, x)

                x_new, dx, _ = env.step(u, noise_override=shared_noise[m, k])

                zs_list.append(np.concatenate([x, u]))
                ys_list.append(dx / dt)
                states.append(x.copy())
                controls.append(u.copy())
                x = x_new

            cost = episode_cost(states, controls, Q, R, dt)
            tax = (
                agent.sigma_u**2 * float(np.trace(R)) * exp_config.T
                if isinstance(agent, _EXCITATION_TYPES)
                else 0.0
            )

            if name != "oracle":
                zs = np.stack(zs_list)
                ys = np.stack(ys_list)
                buffers[name].add_episode(zs, ys)
                agent.update(buffers[name])

            buf = buffers.get(name)
            diag = (
                collect_diagnostics(
                    agent,
                    buf,
                    A_star,
                    B_star,
                    supports,
                    Q,
                    R,
                    threshold=exp_config.support_threshold,
                )
                if buf is not None
                else {}
            )

            if name != "oracle" and m in checkpoint_set and agent.A_est is not None:
                diag["A_est"] = agent.A_est.copy()
                diag["B_est"] = agent.B_est.copy()

            result.episodes[name].append(
                EpisodeRecord(cost=cost, excitation_tax=tax, diagnostics=diag)
            )

    return result


def run_benchmark(exp_config, seeds=None, verbose=False):
    """Run a full benchmark across multiple seeds."""
    if seeds is None:
        seeds = list(range(exp_config.n_seeds))
    results = []
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"Seed {i + 1}/{len(seeds)} (seed={seed})")
        results.append(run_paired_experiment(exp_config, seed, verbose=verbose))
    return results
