"""
Development dashboards.
"""
from __future__ import annotations

import numpy as np
import matplotlib

from common import ExperimentConfig
from metrics import (
    aggregate_trajectory,
    basin_entry_ratio,
    cost_trajectories,
    cumulative_regret_trajectories,
    mean_and_ci,
    per_episode_regret_trajectories,
)

matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm}",
    }
)


def plot_trajectories(results, exp_config: ExperimentConfig, save_path=None):
    import matplotlib.pyplot as plt

    ALL_AGENTS = list(exp_config.agents)
    LEARNING_AGENTS = [a for a in ALL_AGENTS if a != "oracle"]
    COLORS = {
        "oracle": "green",
        "dense_greedy": "blue",
        "dense_excited": "purple",
        "sparse_greedy": "red",
        "sparse_excited": "orange",
    }
    LABELS = {
        "oracle": "Oracle",
        "dense_greedy": "Dense-Greedy",
        "dense_excited": "Dense-Excited",
        "sparse_greedy": "Sparse-Greedy",
        "sparse_excited": "Sparse-Excited",
    }

    M = exp_config.max_episodes
    episodes = np.arange(1, M + 1)

    # (title, key, y_scale, agent_list)
    PANELS = [
        # Row 0
        (r"Cumulative Regret $R_m$", "cumul_regret", "lin", ALL_AGENTS),
        (r"Cumulative Regret $R_m$ (exp)", "cumul_regret", "exp", ALL_AGENTS),
        (r"Per-episode Regret $r_m$", "per_ep_regret", "lin", ALL_AGENTS),
        (r"Episode Cost $J(\bm{\pi}_m)$", "episode_cost", "lin", ALL_AGENTS),
        # Row 1
        (
            r"Parameter Error in $\mathbf{\Theta}$ (log)",
            "error_joint",
            "log",
            LEARNING_AGENTS,
        ),
        (r"Parameter Error in $\mathbf{A}$ (log)", "error_A", "log", LEARNING_AGENTS),
        (r"Parameter Error in $\mathbf{B}$ (log)", "error_B", "log", LEARNING_AGENTS),
        (
            r"Spectral Abscissa $\max \mathrm{Re}(\lambda(\mathbf{A}_\star + \mathbf{B}_\star \mathbf{K}_m(0)))$",
            "spectral_abscissa_t0",
            "lin",
            LEARNING_AGENTS,
        ),
        # Row 2
        (
            r"Support F1 in $\mathbf{\Theta}$",
            "support_f1_joint",
            "lin",
            LEARNING_AGENTS,
        ),
        (r"Support F1 in $\mathbf{A}$", "support_f1_A", "lin", LEARNING_AGENTS),
        (r"Support F1 in $\mathbf{B}$", "support_f1_B", "lin", LEARNING_AGENTS),
        (
            r"Gram Min Eigenvalue $\min_i \lambda_{\min}(\mathbf{Z}_{S_i}^\top \mathbf{Z}_{S_i}/N_m)$",
            "gram_min_eig",
            "lin",
            LEARNING_AGENTS,
        ),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)

    fig.suptitle(
        f"$d={exp_config.system.d}$, $p={exp_config.system.p}$, "
        f"$s={exp_config.system.sparsity}$, $M={exp_config.max_episodes}$, "
        f"$T={exp_config.system.T}$, "
        + r"$\mathrm{d}t$="
        + f"{exp_config.system.dt}, "
        + r"$\sigma_x=$"
        + f"{exp_config.system.sigma}, "
        + r"$\sigma_u=$"
        + f"{exp_config.excitation.sigma_u}, "
        + (
            (r"$c_\lambda=$" + f"{exp_config.estimators.c_lambda}, ")
            if exp_config.estimators.lambda_lasso is None
            else ""
        )
        + (
            f"$\lambda={exp_config.estimators.lambda_lasso}$, "
            if exp_config.estimators.lambda_lasso is not None
            else ""
        )
        + f"$\mu={exp_config.estimators.mu_ridge}$, "
        + f"{exp_config.n_seeds} seeds",
        fontsize=11,
    )

    for ax, panel in zip(axes.flat, PANELS):
        title, key, y_scale, agents = panel

        for name in agents:
            if name not in COLORS:
                continue

            # Data extraction
            if key == "cumul_regret":
                data = cumulative_regret_trajectories(results, name, "oracle")
            elif key == "per_ep_regret":
                if name == "oracle":
                    if y_scale == "exp":
                        data = np.ones((len(results), M))
                    else:
                        data = np.zeros((len(results), M))
                else:
                    data = per_episode_regret_trajectories(results, name, "oracle")
            elif key == "episode_cost":
                data = cost_trajectories(results, name)
            else:
                data = aggregate_trajectory(results, name, key)

            if data.size == 0 or np.all(np.isnan(data)):
                continue

            # Plotting lines only where we have valid (non-NaN) data
            valid_mask = ~np.isnan(data[0])
            valid_episodes = episodes[valid_mask]
            if len(valid_episodes) == 0:
                continue

            valid_data = data[:, valid_mask]
            mean, ci_lo, ci_hi = mean_and_ci(valid_data, axis=0)

            ax.plot(
                valid_episodes,
                mean,
                color=COLORS[name],
                label=LABELS[name],
                linewidth=1.6,
            )
            ax.fill_between(
                valid_episodes,
                ci_lo,
                ci_hi,
                color=COLORS[name],
                alpha=0.15,
            )

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.tick_params(labelsize=7)
        if y_scale == "log":
            ax.set_yscale("log", nonpositive="mask")
        elif y_scale == "exp":
            ax.set_yscale(
                "function",
                functions=(
                    lambda x: np.exp(x),
                    lambda x: np.log(np.clip(x, 1.0, None)),
                ),
            )

    axes[0, 0].legend(fontsize=7, loc="upper left")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_basin_entry_comparison(results, exp_config: ExperimentConfig, save_path=None):
    import matplotlib.pyplot as plt

    available = set(results[0].agent_names) if results else set()
    dense_name, sparse_name = "dense_greedy", "sparse_greedy"

    if dense_name not in available or sparse_name not in available:
        return

    # Threshold = max of mean final parameter error across learning agents.
    # We ask "when did each agent first reach the accuracy
    # level that the worst agent achieves at the end of training?"
    learning_agents = [a for a in exp_config.agents if "greedy" in a and a in available]
    final_errors = []
    for name in learning_agents:
        traj = aggregate_trajectory(results, name, "error_joint")
        valid_finals = [float(v) for v in traj[:, -1] if np.isfinite(v)]
        if valid_finals:
            final_errors.append(float(np.mean(valid_finals)))
    threshold = float(max(final_errors)) if final_errors else 0.3

    ratios, median, stats = basin_entry_ratio(
        results, dense_name, sparse_name, threshold=threshold
    )
    theoretical = exp_config.theoretical_speedup

    fig, ax = plt.subplots(figsize=(6, 4))
    valid = ratios[np.isfinite(ratios)]
    if len(valid) > 0:
        # A (near-)zero-width range breaks fixed-count binning
        degenerate = np.ptp(valid) <= 1e-9 * max(1.0, np.max(np.abs(valid)))
        ax.hist(valid, bins=1 if degenerate else 20, alpha=0.7, label="Empirical ratios")
        ax.axvline(
            median, color="blue", linestyle="--", label=f"Median: {float(median):.2f}"
        )
    ax.axvline(
        theoretical,
        color="red",
        linestyle="-",
        label=f"Theory: {float(theoretical):.2f}",
    )
    ax.set_xlabel(r"$m_0^{\mathrm{dense}} / m_0^{\mathrm{sparse}}$")
    ax.set_ylabel("Count")
    # Surface the censoring: a high dense-never-entered rate means the median is
    # a conservative lower bound (those seeds are capped at M+1, not dropped).
    ax.set_title(
        f"Basin Entry Speedup ($\\epsilon={threshold:.3f}$)\n"
        f"never entered -- dense: {stats['dense_never_entered']:.0%}, "
        f"sparse: {stats['sparse_never_entered']:.0%}"
    )
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _build_true_support_mask(d, p, true_supports, block):
    if block == "A":
        mask = np.zeros((d, d), dtype=bool)
        for i, supp in enumerate(true_supports):
            for j in supp:
                if j < d:
                    mask[i, j] = True
    else:
        mask = np.zeros((d, p), dtype=bool)
        for i, supp in enumerate(true_supports):
            for j in supp:
                if j >= d:
                    mask[i, j - d] = True
    return mask


def _draw_support_overlay(ax, mask):
    import matplotlib.patches as mpatches

    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        ax.add_patch(
            mpatches.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                linewidth=1.2,
                edgecolor="black",
                facecolor="none",
            )
        )


def plot_parameter_evolution(results, exp_config: ExperimentConfig, output_dir):
    """
    For the seed with the best mean final support F1 across learning agents, produce
    two sets of figures per matrix block:

    - params_{block}_best_seed.png : estimated parameter matrices at checkpoints
                                     (RdBu_r, diverging around zero)
    - error_{block}_best_seed.png  : |estimate − truth| at checkpoints
                                     (Reds, starting from zero)
    """
    import matplotlib.pyplot as plt
    import os

    AGENT_LABELS = {
        "dense_greedy": "Dense",
        "dense_excited": "Dense-Ex",
        "sparse_greedy": "Sparse-Gr",
        "sparse_excited": "Sparse-Ex",
    }
    LEARNING_AGENTS = [a for a in exp_config.agents if a in AGENT_LABELS]

    d, p, M = exp_config.system.d, exp_config.system.p, exp_config.max_episodes

    # Select the seed with the highest mean final support F1 across learning agents.
    best_idx, best_score = 0, -np.inf
    for idx, r in enumerate(results):
        scores = []
        for name in LEARNING_AGENTS:
            traj = r.diagnostic_trajectory(name, "support_f1_joint")
            finite = [v for v in traj if np.isfinite(v)]
            if finite:
                scores.append(finite[-1])
        if scores:
            score = float(np.mean(scores))
            if score > best_score:
                best_score, best_idx = score, idx

    n_checkpoints = min(8, M)
    checkpoint_episodes = sorted(
        set(np.round(np.linspace(0, M - 1, n_checkpoints)).astype(int).tolist())
    )

    result = results[best_idx]
    seed = result.seed
    A_true, B_true, supports = result.A_star, result.B_star, result.supports
    vmax = float(np.max(np.abs(np.hstack([A_true, B_true]))))

    mask_A = _build_true_support_mask(d, p, supports, "A")
    mask_B = _build_true_support_mask(d, p, supports, "B")

    shared_suptitle = (
        f"best seed (seed {seed}, F1={best_score:.2f}) — "
        f"d={d}, p={p}, s={exp_config.system.sparsity}, M={M}"
    )

    for block, true_mat, mask, ncols_matrix in [
        ("A", A_true, mask_A, d),
        ("B", B_true, mask_B, p),
    ]:
        n_cols = 1 + len(checkpoint_episodes)
        n_rows = len(LEARNING_AGENTS)
        fig_w = min(n_cols * ncols_matrix * 0.35 + n_cols * 0.15 + 1.5, 28.0)
        fig_h = min(n_rows * d * 0.35 + n_rows * 0.15 + 1.0, 20.0)

        # Each view tuple: (col0_label, col0_mat, est_mat_fn, cmap, vmin, filename, title_prefix)
        VIEWS = [
            (
                "True",
                true_mat,
                lambda est, t=true_mat: est,
                "RdBu_r",
                -vmax,
                f"params_{block}_best_seed.png",
                f"{block} block: estimated parameters",
            ),
            (
                f"|{block}*|",
                np.abs(true_mat),
                lambda est, t=true_mat: np.abs(est - t),
                "Reds",
                0.0,
                f"error_{block}_best_seed.png",
                f"{block} block: |estimate \u2212 truth|",
            ),
        ]

        for (
            col0_label,
            col0_mat,
            est_mat_fn,
            cmap,
            vmin,
            filename,
            title_prefix,
        ) in VIEWS:
            col_titles = [col0_label] + [f"Ep. {m + 1}" for m in checkpoint_episodes]

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                constrained_layout=True,
            )

            for row_idx, agent_name in enumerate(LEARNING_AGENTS):
                for col_idx in range(n_cols):
                    ax = axes[row_idx, col_idx]
                    if col_idx == 0:
                        mat = col0_mat
                    else:
                        ep = result.episodes[agent_name][
                            checkpoint_episodes[col_idx - 1]
                        ]
                        est = ep.diagnostics.get(f"{block}_est", None)
                        if est is None:
                            ax.set_visible(False)
                            continue
                        mat = est_mat_fn(est)

                    ax.imshow(
                        mat,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        aspect="auto",
                        interpolation="nearest",
                    )
                    _draw_support_overlay(ax, mask)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines[:].set_visible(False)

                    if row_idx == 0:
                        ax.set_title(col_titles[col_idx], fontsize=8, pad=3)
                    if col_idx == 0:
                        ax.set_ylabel(
                            AGENT_LABELS[agent_name],
                            fontsize=8,
                            rotation=90,
                            labelpad=4,
                        )

            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[:, -1], shrink=0.6, pad=0.02, aspect=20)
            cbar.ax.tick_params(labelsize=7)

            fig.suptitle(f"{title_prefix} — {shared_suptitle}", fontsize=9, y=1.01)
            fig.savefig(
                os.path.join(output_dir, filename), dpi=120, bbox_inches="tight"
            )
            plt.close(fig)


def plot_self_exploration_diagnostics(
    results: list,
    exp_config: ExperimentConfig,
    save_path: str = None,
) -> None:
    """
    Two-panel diagnostic for the self-exploration condition (Basei et al. 2022, Prop 2.1).

    Left:  Scatter of lambda_min(B*^T Q B*) vs final cumulative regret per seed,
           one series per learning agent. Vertical dashed line at zero marks the
           boundary of the sufficient condition for identifiability.

    Right: Histogram of lambda_min across seeds with the same reference line.
    """
    import matplotlib.pyplot as plt
    import os

    COLORS = {
        "dense_greedy": "blue",
        "dense_excited": "purple",
        "sparse_greedy": "red",
        "sparse_excited": "orange",
    }
    LABELS = {
        "dense_greedy": "Dense-Greedy",
        "dense_excited": "Dense-Excitation",
        "sparse_greedy": "Sparse-Greedy",
        "sparse_excited": "Sparse-Excitation",
    }
    MARKERS = {
        "dense_greedy": "o",
        "dense_excited": "s",
        "sparse_greedy": "^",
        "sparse_excited": "D",
    }

    learning_agents = [a for a in exp_config.agents if a != "oracle" and a in COLORS]
    min_eigs = np.array([r.btqb_min_eig for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter min_eig vs final adjusted regret
    ax = axes[0]
    for name in learning_agents:
        final_regrets = np.array([r.cumulative_regret(name)[-1] for r in results])
        ax.scatter(
            min_eigs,
            final_regrets,
            color=COLORS[name],
            marker=MARKERS[name],
            label=LABELS[name],
            alpha=0.75,
            s=45,
            zorder=3,
        )
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label=r"$\lambda_{\min} = 0$",
    )
    ax.set_xlabel(
        r"$\lambda_{\min}(\mathbf{B}_\star^\top \mathbf{Q} \, \mathbf{B}_\star)$"
    )
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("Self-exploration condition vs regret")
    ax.legend(fontsize=8, framealpha=0.7)

    # Right: histogram of min_eig
    ax = axes[1]
    n_bins = max(10, min(30, len(results) // 3))
    finite_eigs = min_eigs[np.isfinite(min_eigs)]
    # A (near-)zero-width range breaks fixed-count binning ("Cannot create N finite-sized
    # bins") — e.g. ieee39's system is identical across seeds, or a normalised BᵀQB whose
    # λ_min is 1 up to float noise. Use a single bin when the spread is negligible.
    if finite_eigs.size == 0 or np.ptp(finite_eigs) <= 1e-9 * max(1.0, np.max(np.abs(finite_eigs))):
        n_bins = 1
    ax.hist(
        finite_eigs,
        bins=n_bins,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        label=r"$\lambda_{\min} = 0$",
    )
    ax.set_xlabel(
        r"$\lambda_{\min}(\mathbf{B}_\star^\top \mathbf{Q} \, \mathbf{B}_\star)$"
    )
    ax.set_ylabel("Count")
    ax.set_title(r"Distribution of $\lambda_{\min}$ across seeds")
    ax.legend(fontsize=8, framealpha=0.7)

    fig.suptitle(
        rf"Self-exploration diagnostics — "
        rf"$d={exp_config.system.d}$, $p={exp_config.system.p}$, "
        rf"$s={exp_config.system.sparsity}$, $N={{{len(results)}}}$ seeds",
        fontsize=11,
    )
    fig.tight_layout()

    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
