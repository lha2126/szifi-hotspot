#!/bin/bash

#SBATCH --job-name=cluster_analysis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --time=07:58:59
#SBATCH -N 1
#SBATCH --cpus-per-task 1
#SBATCH -p genx
#SBATCH --array=0-77
#SBATCH --output=/mnt/home/ophilcox/out/cluster_analysis%a.log

# Load modules
conda init
conda activate nmt-env

python -u injected_analysis_sep.py $SLURM_ARRAY_TASK_ID 0.00
python -u injected_analysis_sep.py $SLURM_ARRAY_TASK_ID 5.00
python -u injected_analysis_sep.py $SLURM_ARRAY_TASK_ID 10.00
python -u injected_analysis_sep.py $SLURM_ARRAY_TASK_ID 20.00

python -u injected_analysis.py $SLURM_ARRAY_TASK_ID 0.00
python -u injected_analysis.py $SLURM_ARRAY_TASK_ID 5.00
python -u injected_analysis.py $SLURM_ARRAY_TASK_ID 10.00
python -u injected_analysis.py $SLURM_ARRAY_TASK_ID 20.00