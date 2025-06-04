#!/bin/bash

#SBATCH --job-name=generate_couplings
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --time=07:58:59
#SBATCH -N 1
#SBATCH --cpus-per-task 1
#SBATCH -p genx
#SBATCH --array=0-77
#SBATCH --output=/mnt/home/ophilcox/out/generate_couplings%a.log

# Load modules
conda init
conda activate nmt-env

python -u generate_couplings_sep.py $SLURM_ARRAY_TASK_ID