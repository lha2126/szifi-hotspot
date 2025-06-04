#!/bin/bash

#SBATCH --job-name=run_clusters
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --time=07:58:59
#SBATCH -N 1
#SBATCH --cpus-per-task 1
#SBATCH -p genx
#SBATCH --array=0-77
#SBATCH --output=/mnt/home/ophilcox/out/run_clusters%a.log

# Load modules
conda init
conda activate nmt-env

python -u run_clusters.py $SLURM_ARRAY_TASK_ID