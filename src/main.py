"""
Top-level script for running benchmarks defined in configs/.

Benchmark configs: configs/benchmarks/<name>.yaml
Sweep configs:     configs/sweeps/<name>.yaml

Usage:
    python main.py                        # all benchmarks
    python main.py --benchmark d20        # single benchmark
    python main.py --sweep excitation     # single sweep
    python main.py --quick                # quick smoke test
"""

import argparse
import dataclasses
import json
import os
import time
import glob

import numpy as np
import yaml

from common import ExperimentConfig
from runner import run_benchmark
from analysis import (
    all_pairwise_tests,
    final_summary_table,
    print_summary,
    plot_trajectories,
    plot_basin_entry_comparison,
    plot_sparsity_evolution,
    plot_error_evolution,
    basin_entry_ratio,
    seed_wins,
)

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
)

# Config loading


def load_all_benchmarks() -> dict:
    benchmarks = {}
    pattern = os.path.join(CONFIG_DIR, "benchmarks", "*.yaml")
    for path in sorted(glob.glob(pattern)):
        name = os.path.splitext(os.path.basename(path))[0]
        benchmarks[name] = ExperimentConfig.from_yaml(path)
    return benchmarks


def load_benchmark(name: str) -> ExperimentConfig:
    path = os.path.join(CONFIG_DIR, "benchmarks", f"{name}.yaml")
    return ExperimentConfig.from_yaml(path)


def load_sweep(name: str) -> dict:
    """
    Load a sweep config (Option B: base config + list of override dicts).

    Sweep YAML schema:
        base:         <benchmark name>       # base config to load
        name_prefix:  <str>                  # prefix for generated names
        vary:
          - {field1: val1, field2: val2}     # each dict = one override set
          - ...
    """
    path = os.path.join(CONFIG_DIR, "sweeps", f"{name}.yaml")
    with open(path) as f:
        spec = yaml.safe_load(f)

    base = load_benchmark(spec["base"])
    prefix = spec["name_prefix"]
    configs = {}

    for override in spec["vary"]:
        suffix = "_".join(f"{k}{v}" for k, v in override.items())
        key = f"{prefix}_{suffix}"
        configs[key] = ExperimentConfig.apply_overrides(base, override)

    return configs


# Reporting


def run_and_report(name: str, exp_config: ExperimentConfig, output_dir: str):
    """Run one benchmark and save all outputs."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(
        f"  d={exp_config.system.x_dim}, p={exp_config.system.u_dim}, s={exp_config.system.sparsity}={exp_config.system.s_A}+{exp_config.system.s_B}=s_A+s_B"
    )
    print(
        f"  M={exp_config.max_episodes}, H={exp_config.system.H}, seeds={exp_config.n_seeds}, m_explore={exp_config.m_explore}"
    )
    print(f"  agents: {list(exp_config.agents)}")
    print(f"  Theoretical speedup: {exp_config.theoretical_speedup:.2f}")
    print(
        "  Lambda schedule (first 5): "
        + ", ".join(
            f"{exp_config.theoretical_lambda(m * exp_config.system.H):.4f}"
            for m in range(1, 6)
        )
    )
    print(f"{'=' * 60}")
    print(exp_config)
    print(f"{'=' * 60}")

    t0 = time.time()
    results = run_benchmark(exp_config, verbose=True)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    table = final_summary_table(results, agent_names=exp_config.agents)
    print("\nFinal-episode summary:")
    print_summary(table)

    tests = all_pairwise_tests(results)
    print("\nPaired tests:")
    for label, test_dict in tests.items():
        sign = test_dict["sign_test"]
        print(
            f"  {label}: wins={sign['wins_b']}/{sign['n']}, p={sign['p_value']:.4f} (sign test)"
        )

    # Seed wins
    learning = [n for n in exp_config.agents if n != "oracle"]
    if "dense_greedy" in exp_config.agents:
        for sp in [n for n in learning if n != "dense_greedy"]:
            w = seed_wins(results, "dense_greedy", sp)
            print(f"  Seed wins ({sp} < dense_greedy): {w}/{len(results)}")

    ratios, median = basin_entry_ratio(results, threshold=0.15)

    # Save outputs
    bench_dir = os.path.join(output_dir, name, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(bench_dir, exist_ok=True)

    plot_trajectories(
        results, exp_config, save_path=os.path.join(bench_dir, "trajectories.png")
    )
    plot_basin_entry_comparison(
        results, exp_config, save_path=os.path.join(bench_dir, "basin_entry.png")
    )
    param_dir = os.path.join(bench_dir, "params_evolution")
    os.makedirs(param_dir, exist_ok=True)
    plot_sparsity_evolution(results, exp_config, output_dir=param_dir)
    plot_error_evolution(results, exp_config, output_dir=param_dir)

    # Serialise config for reproducibility
    config_dict = {
        f.name: getattr(exp_config, f.name) for f in dataclasses.fields(exp_config)
    }
    # Convert non-JSON-serialisable types
    from dataclasses import is_dataclass

    for k, v in config_dict.items():
        if isinstance(v, tuple):
            config_dict[k] = list(v)
        if is_dataclass(v):
            config_dict[k] = {f.name: getattr(v, f.name) for f in dataclasses.fields(v)}

    save_dict = {
        "config": config_dict,
        "summary": {
            n: {k: list(v) for k, v in row.items()} for n, row in table.items()
        },
        "tests": {
            label: {
                test_name: {
                    k: float(v)
                    if isinstance(v, (int, float, np.floating, np.integer))
                    else v
                    for k, v in test_result.items()
                }
                for test_name, test_result in test_dict.items()
            }
            for label, test_dict in tests.items()
        },
        "basin_entry_median": float(median) if np.isfinite(median) else None,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(bench_dir, "results.json"), "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Results saved to {bench_dir}/")
    return results


# Entry point


def main():
    parser = argparse.ArgumentParser(
        description="Run sparse continuous-time LQ control experiments."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Run a single benchmark by YAML name (e.g. 'd20').",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Run a single sweep by YAML name (e.g. 'excitation').",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run only the quick smoke test."
    )
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick:
        configs = {"quick": load_benchmark("quick")}
    elif args.benchmark:
        configs = {args.benchmark: load_benchmark(args.benchmark)}
    elif args.sweep:
        configs = load_sweep(args.sweep)
    else:
        configs = load_all_benchmarks()
        # Exclude quick from batch runs
        configs = {k: v for k, v in configs.items() if k != "quick"}

    for name, cfg in configs.items():
        run_and_report(name, cfg, args.output_dir)


if __name__ == "__main__":
    main()
