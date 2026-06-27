"""
Cross-seed aggregation, statistical tests, basin-entry analysis, and summary tables
over experiment results.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats


def aggregate_metric(results, agent_name, metric_fn):
    """Extract a per-seed scalar metric and return array over seeds."""
    return np.array([metric_fn(r, agent_name) for r in results])


def aggregate_trajectory(results, agent_name, key):
    """Extract a per-episode diagnostic trajectory for each seed."""
    trajs = [r.diagnostic_trajectory(agent_name, key) for r in results]
    return np.array(trajs)


def cumulative_regret_trajectories(results, agent_name, oracle_name="oracle"):
    """Cumulative regret trajectories, shape (n_seeds, M)."""
    return np.array([r.cumulative_regret(agent_name, oracle_name) for r in results])


def cost_trajectories(results, agent_name):
    """Per-episode cost trajectories across seeds."""
    return np.array([[ep.cost for ep in r.episodes[agent_name]] for r in results])


def per_episode_regret_trajectories(results, agent_name, oracle_name="oracle"):
    """Per-episode cost excess r_m = J_m^agent - J_m^oracle, shape (n_seeds, M)."""
    return cost_trajectories(results, agent_name) - cost_trajectories(
        results, oracle_name
    )


def mean_and_ci(arr, axis=0, confidence=0.95):
    """Compute mean and confidence interval across axis."""
    n = arr.shape[axis]
    mean = np.mean(arr, axis=axis)
    se = np.std(arr, axis=axis, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    return mean, ci_low, ci_high


# ─────────────────────────────────────────────────────────────────────
# Statistical tests (all paired by seed)
# ─────────────────────────────────────────────────────────────────────


def _final_regret(results, agent_name, oracle_name="oracle"):
    """Per-seed final cumulative regret."""
    return cumulative_regret_trajectories(results, agent_name, oracle_name)[:, -1]


def paired_t_test(results, agent_a, agent_b):
    """Two-sided paired t-test on final cumulative regret."""
    vals_a = _final_regret(results, agent_a)
    vals_b = _final_regret(results, agent_b)
    diffs = vals_a - vals_b
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "mean_diff": float(np.mean(diffs)),
    }


def wilcoxon_test(results, agent_a, agent_b, alternative="greater"):
    """One-sided Wilcoxon signed-rank test on final cumulative regret."""
    vals_a = _final_regret(results, agent_a)
    vals_b = _final_regret(results, agent_b)
    diffs = vals_a - vals_b
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2:
        return {"statistic": np.nan, "p_value": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_val = stats.wilcoxon(nonzero, alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p_val)}


def sign_test(results, agent_a, agent_b):
    """Exact two-sided sign test on final cumulative regret."""
    vals_a = _final_regret(results, agent_a)
    vals_b = _final_regret(results, agent_b)
    diffs = vals_a - vals_b
    wins_a = int(np.sum(diffs > 0))
    wins_b = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))
    n = wins_a + wins_b
    if n == 0:
        return {"wins_a": 0, "wins_b": 0, "ties": ties, "n": 0, "p_value": 1.0}
    k = max(wins_a, wins_b)
    p_val = min(2 * stats.binom.sf(k - 1, n, 0.5), 1.0)
    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties, "n": n, "p_value": p_val}


def all_pairwise_tests(results, dense_name="dense_greedy", sparse_names=None):
    if sparse_names is None:
        sparse_names = ["sparse_greedy", "sparse_excited"]
    available = set(results[0].agent_names) if results else set()
    output = {}
    for sp in sparse_names:
        if dense_name not in available or sp not in available:
            continue
        label = f"{dense_name}_vs_{sp}"
        output[label] = {
            "t_test": paired_t_test(results, dense_name, sp),
            "wilcoxon": wilcoxon_test(results, dense_name, sp),
            "sign_test": sign_test(results, dense_name, sp),
        }
    return output


# ─────────────────────────────────────────────────────────────────────
# Non-endpoint robustness
# ─────────────────────────────────────────────────────────────────────


def seed_wins(results, agent_a, agent_b, oracle_name="oracle"):
    """Seeds where agent_b has lower final cumulative regret than agent_a."""
    return int(
        np.sum(
            _final_regret(results, agent_b, oracle_name)
            < _final_regret(results, agent_a, oracle_name)
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Basin entry analysis
# ─────────────────────────────────────────────────────────────────────


def basin_entry_analysis(
    results, agent_name, thresholds=(0.05, 0.10, 0.15, 0.20, 0.30)
):
    def basin_entry_episode(error_trajectory, thresholds):
        result = {}
        for eps in thresholds:
            try:
                result[eps] = min(
                    m for m, err in enumerate(error_trajectory) if err <= eps
                )
            except ValueError:
                result[eps] = None
        return result

    output = {eps: [] for eps in thresholds}
    for r in results:
        error_traj = r.diagnostic_trajectory(agent_name, "error_joint")
        entries = basin_entry_episode(error_traj, thresholds)
        for eps in thresholds:
            val = entries[eps]
            output[eps].append(val if val is not None else np.nan)
    return {eps: np.array(vals) for eps, vals in output.items()}


def basin_entry_ratio(
    results, dense_name="dense_greedy", sparse_name="sparse_greedy", threshold=0.15
):
    dense_m0 = basin_entry_analysis(results, dense_name, (threshold,))[threshold]
    sparse_m0 = basin_entry_analysis(results, sparse_name, (threshold,))[threshold]

    M = len(results[0].diagnostic_trajectory(dense_name, "error_joint")) if results else 0
    dense_censored = np.isnan(dense_m0)
    sparse_censored = np.isnan(sparse_m0)

    # Episodes-to-enter, 1-indexed; a non-entry is censored at M+1 (a lower bound:
    # M episodes were not enough, so the true entry time is at least M+1).
    dense_time = np.where(dense_censored, M + 1, dense_m0 + 1)
    sparse_time = np.where(sparse_censored, M + 1, sparse_m0 + 1)

    ratios = dense_time / sparse_time
    both_censored = dense_censored & sparse_censored
    ratios[both_censored] = np.nan  # neither entered -> uninformative

    median = float(np.nanmedian(ratios)) if np.any(~both_censored) else np.nan
    n = len(ratios)
    stats = {
        "n_seeds": int(n),
        "dense_never_entered": float(np.mean(dense_censored)) if n else float("nan"),
        "sparse_never_entered": float(np.mean(sparse_censored)) if n else float("nan"),
        "both_never_entered": int(np.sum(both_censored)),
    }
    return ratios, median, stats


# ─────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────


def final_summary_table(results, agent_names, oracle_name="oracle"):
    agent_names = [agent for agent in agent_names if agent != oracle_name]

    def _stats(vals):
        n = len(vals)
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        t_crit = float(stats.t.ppf(0.975, df=max(n - 1, 1)))
        return (m, t_crit * se)

    table = {}
    for name in agent_names:
        row = {
            "final_regret": _stats(_final_regret(results, name, oracle_name)),
        }
        for key in [
            "error_joint",
            "error_A",
            "error_B",
            "support_f1_joint",
            "support_f1_A",
            "support_f1_B",
        ]:
            traj = aggregate_trajectory(results, name, key)
            # Remove NaNs representing missing data before the checkpoint
            valid_final_vals = [val for val in traj[:, -1] if np.isfinite(val)]
            row[key] = (
                _stats(valid_final_vals) if valid_final_vals else (np.nan, np.nan)
            )
        table[name] = row
    return table


def print_summary(table):
    header = (
        f"{'Agent':<22} {'Regret':>14} {'Err(joint)':>12}"
        f"{'Err(A)':>12} {'Err(B)':>12} {'F1(joint)':>12} {'F1(A)':>12} {'F1(B)':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, row in table.items():

        def fmt(key):
            m, ci = row[key]
            return f"{float(m):.2f}±{float(ci):.2f}"

        print(
            f"{name:<22} {fmt('final_regret'):>14} "
            f"{fmt('error_joint'):>12} {fmt('error_A'):>12} {fmt('error_B'):>12} "
            f"{fmt('support_f1_joint'):>12} {fmt('support_f1_A'):>12} {fmt('support_f1_B'):>12}"
        )
