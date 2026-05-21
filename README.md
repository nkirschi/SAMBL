# Sample-Efficient Continuous-Time Reinforcement Learning

## Directory Structure

```
.
├── configs
│   ├── benchmarks
│   │   ├── d100.yaml
│   │   ├── debug.yaml
│   │   └── ...
│   └── sweeps
│       ├── sparsity.yaml
│       └── ...
├── notebooks
├── results
│   ├── benchmarks
│   │   ├── d100
│   │   ├── debug
│   │   └── ...
│   └── sweeps
│       ├── sparsity
│       └── ...
├── slurm
│   ├── stderr
│   └── stdout
├── src
└── tests
```

## Running Experiments (Slurm)

benchmarks:
```
sbatch --cpus-per-task=${NUM_CPUS} job.slurm benchmark ${NAME}
```
sweeps:
```
sbatch --cpus-per-task=${NUM_CPUS} job.slurm sweep ${NAME}
```