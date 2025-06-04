#!/bin/bash

#SBATCH --job-name=pair_template
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --time=23:58:59
#SBATCH -N 1
#SBATCH -p cca
#SBATCH --output=/mnt/home/ophilcox/out/pair_template.log

# Load modules
conda init
conda activate nmt-env

python -u pair_template.py