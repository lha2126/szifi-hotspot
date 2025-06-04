#!/bin/bash

#SBATCH --job-name=multi_cluster_analysis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ophilcox@princeton.edu
#SBATCH --time=23:58:59
#SBATCH -N 1
#SBATCH -p cca
#SBATCH --array=0-7
#SBATCH --output=/mnt/home/ophilcox/out/multi_cluster_analysis%a.log

# Load modules
conda init
conda activate nmt-env

# for i in {0..77}
# do
#    echo "Analysis slice $i"
#    python -u injected_analysis_test2.py $i $SLURM_ARRAY_TASK_ID &
# done
# wait

for i in {0..99}
do
   echo "Analysis slice $i"
   python -u cv_analysis_compsep.py $i $SLURM_ARRAY_TASK_ID &
done
wait
