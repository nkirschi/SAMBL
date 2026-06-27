# Sample-Efficient Continuous-Time Reinforcement Learning

Model-based RL for episodic continuous-time linear-quadratic control where the
dynamics `Θ⋆ = [A⋆ | B⋆]` are **row-sparse**. The experiments show that a row-sparse
(group-LASSO) estimator attains lower regret than dense ridge/OLS as the dimension
grows, on synthetic systems and on two structured benchmarks (a spring-mass chain and
the IEEE 39-bus power grid).

The project uses [`uv`](https://docs.astral.sh/uv/); every command below is prefixed
with `uv run` so it executes inside the locked environment.

## Directory structure

```
.
├── src/                      # all source code
│   ├── main.py               #   CLI entry point: run a benchmark / sweep / debug run
│   ├── runner.py             #   per-seed episode loop, agent construction, RNG handling
│   ├── results_io.py         #   per-seed NPZ + JSON config persistence and loaders
│   ├── common.py             #   config dataclasses (ExperimentConfig, SystemConfig, ...)
│   ├── system_generator.py   #   all system samplers (synthetic, spring chain, IEEE 39-bus)
│   ├── ieee39_data.py        #   baked IEEE 39-bus topology + inertia data (MATPOWER case39)
│   ├── dynamics.py           #   continuous-time SDE simulation
│   ├── agent.py              #   generic linear control agent
│   ├── estimator.py          #   row-LASSO and ridge estimators (incremental Gram)
│   ├── planner.py            #   Riccati solver (Hamiltonian matrix-exponential)
│   ├── diagnostics.py        #   per-episode diagnostics (RE, support recovery, ...)
│   ├── figures.py            #   regenerate all thesis figures from results/
│   ├── metrics.py            #   aggregation + statistics over a result tree
│   └── dashboard.py          #   development plots
├── configs/
│   ├── benchmarks/           # single-system configs (one YAML = one study)
│   └── sweeps/               # parameter sweeps (a base + named `vary` points)
├── notebooks/                # analysis + figure-illustration notebooks
├── results/                  # produced result tree (see "Result layout" below)
├── tests/                    # pytest suite
├── job.slurm                 # SLURM wrapper around src/main.py
└── pyproject.toml            # uv project / dependencies
```

## Running experiments

A study is either a **benchmark** (one system, `configs/benchmarks/<name>.yaml`) or a
**sweep** (a base config plus named override points, `configs/sweeps/<name>.yaml`).

### Locally

```bash
# one benchmark (parallelise across seeds)
uv run python src/main.py --benchmark ieee39 --n-workers 20

# one sweep (parallelise across all (point, seed) pairs)
uv run python src/main.py --sweep synthetic --n-workers 20

# quick smoke run (small d, few seeds)
uv run python src/main.py --debug

# regenerate plots/summaries in place from an existing result dir (no re-simulation)
uv run python src/main.py --replot results/synthetic
```

Flags: `--n-workers N` (process pool; default 1 = serial), `--output-dir DIR`
(default `results`), `--plots` (also render the dev dashboards).

### On the cluster (SLURM)

`job.slurm` forwards its two positional arguments to `--<mode> <name>`:

```bash
sbatch --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} job.slurm benchmark ieee39
sbatch --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} job.slurm sweep synthetic
```

`--n-workers` is set automatically from `--cpus-per-task`.

### Available studies

| kind       | name(s)                                  | what it is                                   |
|------------|------------------------------------------|----------------------------------------------|
| benchmark  | `synthetic`                              | canonical synthetic system (sweep base)      |
| benchmark  | `synthetic_d10` … `synthetic_d100`       | individual synthetic dimension points        |
| benchmark  | `springs_d20`                            | spring-mass chain, d=20                       |
| benchmark  | `ieee39`                                 | IEEE 39-bus power grid (d=78, p=9)            |
| benchmark  | `debug`                                  | tiny config for smoke tests                  |
| sweep      | `synthetic`                              | dimension sweep d ∈ {10,20,50,100}, p=d/2    |
| sweep      | `spring`                                 | dimension sweep on the spring chain          |
| sweep      | `sparsity`                               | fixed d=50, increasing row sparsity s        |
| sweep      | `actuator` `cost` `excitation` `sigmas`  | auxiliary one-parameter sweeps               |

## Result layout

Every run writes a standard per-seed tree (NPZ payload + a JSON config, no pickle):

```
results/<study>/<point>/
    config.json        # the exact ExperimentConfig
    seed_0.npz         # per-agent, per-episode trajectories + diagnostics
    seed_1.npz
    ...
    results.json       # aggregated summary
```

A `--benchmark <name>` writes a single point at `results/<name>/`; a `--sweep <name>`
writes one point per `vary` entry at `results/<name>/<point>/`. Runs resume naturally —
a point directory just accumulates `seed_*.npz`.

## Figures

Regenerate every thesis figure (A–J) from the result tree:

```bash
uv run python src/figures.py            # -> results/figures/*.pdf (+ .png previews)
```

It reads `results/{synthetic,spring,ieee39,sparsity}` (override with
`--synthetic-dir`, `--spring-dir`, `--ieee39-dir`, `--sparsity-dir`, `--out`).

The two system **illustrations** are notebooks that write their PDFs/PNGs into
`notebooks/`:

- `notebooks/spring_chain.ipynb` — spring-mass schematic + `[A⋆|B⋆]` sparsity pattern
- `notebooks/ieee39.ipynb` — grid topology + sparsity pattern

(The other notebooks — `regret_decomposition`, `pure_exploration`, `speedup_ratio`,
`dynamics` — are exploratory analyses.)

## Tests

```bash
uv run pytest -q
```
