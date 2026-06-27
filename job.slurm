#!/bin/sh
#
#SBATCH -J sectrl
#SBATCH --output=/dss/dsshome1/07/go69lir/SECTRL/slurm/sectrl.%j.out
#SBATCH --error=/dss/dsshome1/07/go69lir/SECTRL/slurm/sectrl.%j.err
#
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=16GB
#SBATCH --time=0-00:30:00

# Prevent each worker from spawning its own BLAS thread pool
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PATH="$PATH:/dss/dsshome1/07/go69lir/texlive/2026/bin/x86_64-linux"

MODE="$1 $2"  # e.g., 'benchmark synthetic_d20' or 'sweep sparsity'
PYTHON="/dss/dsshome1/07/go69lir/SECTRL/.venv/bin/python"
N_WORKERS=${SLURM_CPUS_PER_TASK}

echo "Running ${MODE} with ${N_WORKERS} workers"

srun --cpu-bind=cores,verbose \
    ${PYTHON} -u src/main.py \
        --${MODE} \
        --n-workers ${N_WORKERS} \
        --plots