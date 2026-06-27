# CLAUDE.md — SECTRL project guide

Handoff notes for a Claude session continuing this work (incl. on the LRZ cluster).
Read this first; it captures the current state, conventions, and the live workflows.

## What this project is

Master's thesis **"Sample-Efficient Reinforcement Learning under Sparse Dynamics"**
(author: Niki). Model-based RL for **episodic continuous-time linear-quadratic (LQ)
control** where the dynamics matrix `Θ⋆ = [A⋆ | B⋆]` is **row-sparse**. The thesis shows a
row-sparse (group-LASSO) estimator attains lower regret than dense ridge/OLS as the state
dimension `d` grows — on synthetic systems and on two structured real-world-flavoured
benchmarks (a spring-mass chain and the IEEE 39-bus power grid).

Dynamics: `dx = (A⋆ x + B⋆ u) dt + σ dW`, finite horizon `T`, episodic over `M` episodes,
quadratic cost. Speedup intuition: `(d+p)/(s·log(d+p))` (loose worst-case; empirical
advantage is ~1.3–1.6×).

## Environment & how to run

- **`uv` project**, Python 3.10. Run everything via `uv run python src/main.py …`.
- Tests: `uv run pytest -q` (should be **63 passing**).
- Cluster (LRZ): repo at `/dss/dsshome1/07/go69lir/SECTRL`, venv at `.../SECTRL/.venv`.
  `job.slurm` wraps `main.py` (see "Cluster workflow" below).

## Repository layout

```
src/                  # all source (flat package; modules import each other bare, e.g. `from common import …`)
  main.py             #   CLI: --benchmark / --sweep / --debug / --replot, runs + persists + reports
  runner.py           #   per-seed episode loop (matched seeds), agent construction, family dispatch (match), find_result_dirs
  results_io.py       #   per-seed NPZ + JSON config I/O; load_point / load_study  (NPZ ONLY — no pickle)
  common.py           #   config dataclasses (ExperimentConfig + nested System/Cost/Estimator/Excitation), from_yaml
  system_generator.py #   sample_synthetic_system, sample_spring_chain, sample_ieee39 (+ _is_stabilisable/_is_controllable)
  ieee39_data.py      #   baked IEEE 39-bus topology + inertia constants (data only)
  dynamics.py         #   continuous-time SDE env (ContinuousLQREnv)
  estimator.py        #   RowLassoEstimator (coord descent, incremental Gram) + DiscreteRidgeEstimator
  planner.py          #   RiccatiODESolver — exact Hamiltonian matrix-exp Möbius recurrence
  diagnostics.py      #   per-episode diagnostics (param error, support recovery, RE eig, B^T Q B, ...)
  figures.py          #   publication figures A–J -> results/figures/  (reads the NPZ tree)
  metrics.py          #   aggregation + stats over a result tree (basin_entry, etc.)
  dashboard.py        #   dev/diagnostic PNGs (opt-in)
configs/
  benchmarks/<name>.yaml   # single-system studies
  sweeps/<name>.yaml       # base benchmark + named `vary` override points
  sweeps/_archive/         # retired sweeps (may reference OLD bare-dN bases — stale, ignore)
notebooks/            # ieee39.ipynb, spring_chain.ipynb (system illustrations, write PDFs into notebooks/) + analysis nbs
results/              # GITIGNORED produced tree (see "Result layout")
tests/                # pytest suite
job.slurm             # SLURM wrapper around main.py
thesis_latex/         # GITIGNORED, being removed — NO CODE MAY DEPEND ON IT
README.md             # user-facing run guide (kept in sync with this)
```

## The five agents (2×2 + oracle)

| agent | estimator | exploration |
|---|---|---|
| `oracle` | true `Θ⋆` (zero-regret benchmark, optimal time-varying gain) | — |
| `dense_greedy` | ridge `μ=1e-6` (≈ OLS — **intentionally**, fair baseline) | certainty equivalence |
| `dense_excited` | ridge | + isotropic Gaussian probing `σ_u` |
| `sparse_greedy` | **row-LASSO** | certainty equivalence |
| `sparse_excited` | row-LASSO | + `σ_u` |

Dense vs sparse share data/planner/noise and differ only in the estimator, so any gap is
the sparsity penalty alone. **Matched seeds (CRN):** per seed, shared `noise_rng`,
`explore_rng`, `x0_rng` with fixed offsets; per-agent RNGs derived from the seed.

Key internals: estimator accumulates the **Gram `ZᵀZ` incrementally** across episodes
(warm-started coord descent), `λ = theoretical_lambda` (scaled by `c_lambda`) with an
episode-1 warmup. Planner solves the differential Riccati eq in time-to-go via the
**Hamiltonian** `M=[[-A,S],[Q,Aᵀ]]`, `S=BR⁻¹Bᵀ`, `E=exp(M·Δt)`, Möbius recurrence
`P_{k+1}=(E21+E22·P)(E11+E12·P)⁻¹` — exact, constant-time, stiffness-free.

## System families & samplers

`SystemConfig.family ∈ {synthetic, spring_chain, ieee39}` selects the sampler in
`runner.run_paired_experiment` (a `match`; `case _` **raises** on unknown family). All
samplers take **raw parameters** (not a config) and return `(A, B, supports, attempt)`:

- `sample_synthetic_system(d, p, s_A, s_B, seed, a_min, a_max, b_min, b_max, normalise_B=False)`
  — row-sparse; `normalise_B` rescales B columns to unit norm (equal control authority,
  well-conditioned `BᵀQB`; removes a conditioning confound — default **True** in configs).
- `sample_spring_chain(d, p, seed, k_min, k_max, m_min, m_max)` — banded undamped chain,
  `d=2n`, randomized masses/springs per seed, `p` evenly-spaced actuators.
- `sample_ieee39(seed, m_load=10, damping=3, jitter=0)` — swing-oscillator network,
  fixed `d=78, p=9` (bus 39 = unactuated reference); topology/inertia in `ieee39_data.py`
  (sources: MATPOWER `case39`, Athay et al. 1979 / Pai 1989 inertias).

## Result layout & format (NPZ only)

A run writes per study/point:
```
results/<study>/<point>/
  config.json     # exact ExperimentConfig (json)
  seed_<n>.npz    # per-agent per-episode trajectories + diagnostics (+ A_star/B_star/supports)
  results.json    # aggregated summary
```
`--benchmark <name>` → `results/<name>/`. `--sweep <name>` → `results/<name>/<point>/`.
Runs **resume naturally** (a point dir just accumulates `seed_*.npz`). **Legacy pickle
support was removed** — `load_point`/`load_study` read npz only; `config_*_json`,
`seed_*_npz`, `persist_point` are the I/O surface.

## Studies (configs)

**Benchmarks:** `synthetic_d10/d20/d50/d100` (synthetic dimension points),
`springs` (spring chain), `ieee39` (power grid), `synthetic` (canonical sweep base),
`clambda` (sweep base), `debug` (tiny smoke).
**Sweeps:** `synthetic` (d∈{10,20,50,100}, p=d/2), `springs` (spring chain over d),
`sparsity` (fixed d=50, p=25; **9 points** s2…s10), `clambda` (c_λ × d for figure G),
`actuator`/`cost`/`excitation`/`sigmas` (auxiliary, not in the thesis figures).

Sweep YAML = `base: <benchmark>` + `vary: [{name: <point>, <field>: <val>, …}]`. Overrides
map to nested sub-configs automatically (e.g. `c_lambda`→estimators, `d`/`p`→system).

## Figures

```bash
uv run python src/figures.py          # -> results/figures/*.pdf (+ .png previews)
```
Reads `results/{synthetic, springs, ieee39, sparsity, clambda}` (each overridable via
`--*-dir`). Panels: A–F synthetic (`results/synthetic`), G c_λ (`results/clambda`),
H spring (`results/springs`), I ieee39 (`results/ieee39`), J sparsity (`results/sparsity`).
Missing studies are skipped gracefully. usetex is on (needs LaTeX on PATH).

**Dashboards are NOT auto-generated** — they're the dev PNGs in `dashboard.py`, opt-in via
`main.py --plots`, or regenerated for existing dirs via `main.py --replot <path>`. The
thesis figures (`figures.py`, A–J) are separate.

## Cluster workflow — regenerate all results (likely the handoff task)

`job.slurm` forwards two positional args to `main.py --<arg1> <arg2>` and sets
`--n-workers` from `--cpus-per-task`. Defaults: 20 cpus / 16 GB / 30 min. It **does**
forward `--plots`, and it exports the user-local TeX Live onto PATH (batch jobs run
`#!/bin/sh` non-interactively and do **not** source `~/.bashrc`, where that PATH normally
lives), so the dev dashboards render in-batch via matplotlib usetex. Measured wall times
are small: even synthetic `d=100`/`p=50` is ~50 s/seed → ~1.7 min/point at 20 workers, and
every study finishes in well under 20 min at ~1.3 GB peak RAM — so the limits below (1 h,
default 16 GB) carry large margin. Tune from `sacct -j <id> --format=Elapsed,MaxRSS`.

```bash
# from the repo root on the cluster — the five (npz) studies that feed all figures:
sbatch --time=01:00:00 job.slurm sweep     synthetic   # -> results/synthetic/{d10,d20,d50,d100}/
sbatch --time=01:00:00 job.slurm sweep     springs     # -> results/springs/{d10,d20,d50,d100}/
sbatch --time=01:00:00 job.slurm sweep     sparsity    # -> results/sparsity/{s2..s10}/  (9 points)
sbatch --time=01:30:00 job.slurm sweep     clambda     # -> results/clambda/<15 points>/ (figure G)
sbatch --time=00:30:00 job.slurm benchmark ieee39      # -> results/ieee39/
```
Then regenerate figures (after `rsync`-ing results back, or on the cluster):
`uv run python src/figures.py` — needs that same TeX Live on PATH (interactive shells get it
from `~/.bashrc`; a non-interactive one must add `~/texlive/2026/bin/x86_64-linux`).

## Conventions & decisions established this session

- **Samplers take raw params**, not `SystemConfig` (mirrors `sample_synthetic_system`).
- **`family` is explicit** in every config under `system:`; runner dispatch raises on unknown.
- **NPZ is the only result format** (legacy `seed_results.pkl` support fully removed).
- **Config naming:** synthetic dimension benches are `synthetic_dN`; spring is `springs_*`.
- **`sample_sparse_system` was renamed `sample_synthetic_system`.**
- Family dispatch in `runner.run_paired_experiment` is a `match` statement.
- IEEE 39-bus **data lives in `ieee39_data.py`**; the sampler in `system_generator.py`.
- Reporting statistic for figures is **median ± IQR** (regret is heavy-tailed).
- System illustration scripts became **notebooks** (`notebooks/{ieee39,spring_chain}.ipynb`)
  that save figures into `notebooks/` (NOT `thesis_latex/`).

## Gotchas / caveats

- **`thesis_latex/` is gitignored and slated for removal — keep all code free of it.**
  (The chapter-5 text + an "Appendix: Construction of the IEEE 39-Bus Anchor" with proper
  citations live there; the appendix is an unnumbered `\chapter*` because the thesis class
  renders chapter numbers as die faces via `\dice{\thechapter}`, which breaks on appendix
  letters — don't use a numbered appendix chapter.)
- **Figure G (`clambda`)**: `clambda.yaml` = 15 points (3 d × 5 c_λ), agents =
  oracle+sparse_greedy, n_seeds=25. Being an oracle+sparse-only study it yields
  `basin_entry=null` (the dense-vs-sparse metric is undefined; guarded in `_build_save_dict`).
- **Stale junk in `results/`** (gitignored, safe to ignore/delete): old `results/{benchmarks,
  sweeps,d10,d20,hpo,hpo_clambda}/` from the pre-restructure layout, and ~117 now-dead
  `seed_results.pkl` files (nothing reads them anymore — `find results -name seed_results.pkl -delete`).
- `configs/sweeps/_archive/*` still reference old bare-`dN` bases — retired, don't run as-is.
- **Work is uncommitted.** There's a proposed focused-commit sequence (ask Niki / see chat
  history) — `thesis_latex/` changes are out of git scope. Repo commit style: short
  imperative subject, no `type:` prefix.

## Status

- Suite green (63). **All five figure studies were regenerated on the cluster (2026-06-27,
  jobs 5690309–13, all COMPLETED in 4–13 min, ≤2.5 GB RAM) and all 10 figures A–J built**
  into `results/figures/*.pdf` (incl. the previously-never-run G). Getting there fixed four
  bugs (all currently uncommitted): (1) created the missing `configs/benchmarks/synthetic.yaml`
  base that the synthetic+sparsity sweeps reference; (2) job.slurm now exports the user-local
  TeX Live onto PATH so the `--plots` dashboards render in batch (batch shells don't source
  `~/.bashrc`); (3) guarded `basin_entry` for oracle+sparse-only studies (clambda KeyError);
  (4) hardened dashboard histograms against a (near-)zero-width range (ieee39 / normalised
  BᵀQB). Old `results/{benchmarks,sweeps}/` pickle dirs remain (gitignored, safe to delete).
