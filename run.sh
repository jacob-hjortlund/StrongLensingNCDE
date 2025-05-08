#!/bin/bash
#SBATCH -p ampere
#SBATCH -t 2-00:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --mail-user=jacob.hjortlund@fysik.su.se
#SBATCH --job-name=specz_xe_128
#SBATCH --output=/cfs/home/jahj0154/logs/%x_%j.out
#SBATCH --error=/cfs/home/jahj0154/logs/%x_%j.err

export SLURM_CPU_BIND="cores"

# — initialize micromamba’s shell functions
eval "$(micromamba shell hook -s bash)"

# — activate the environment you want
micromamba activate StrongLensingNCDE

srun python /cfs/home/jahj0154/StrongLensingNCDEmodel_training.py