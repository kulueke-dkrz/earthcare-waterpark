#!/bin/bash
#SBATCH --job-name=earthcare_pyramid
#SBATCH --partition=shared               # serial/data-processing jobs go on 'shared', not 'compute'
#SBATCH --mem=32G                        # Slurm auto-derives the CPU count from this (shared: 940 MB/CPU)
#SBATCH --time=1-00:00:00                # shared allows up to 7-00:00:00 if you need more
#SBATCH --mail-type=FAIL
#SBATCH --account=bk1088      # REQUIRED on Levante, e.g. bk1088 - replace with your project account
#SBATCH --output=logs/earthcare_pyramid.o%j
#SBATCH --error=logs/earthcare_pyramid.e%j

set -e
ulimit -s 204800
mkdir -p logs

# Load/activate your Python environment.
# If you use a plain module-provided Python:
# module load python3
# If you use your own conda/mamba environment (as before):
source ~/.bashrc
source activate /home/k/k204228/.conda/envs/earthcare

python -u process_earthcare_day.py \
    --start-date 2024-08-12 \
    --end-date 2024-08-20
