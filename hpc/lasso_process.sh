#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=128gb
#SBATCH --array=1-96%20

ALPHAS=(0.005 0.01 0.02 0.03 0.04 0.05 0.10 0.15)

SCENARIO=$(( (SLURM_ARRAY_TASK_ID - 1) / 8 + 1 ))
ALPHA=${ALPHAS[$(( (SLURM_ARRAY_TASK_ID - 1) % 8 ))]}

PYTHON=/mnt/beegfs/home/juan.vidal/miniconda3/envs/knncpu/bin/python3.8
echo "scenario=$SCENARIO alpha=$ALPHA"
$PYTHON code/lasso/sim_lasso.py -s $SCENARIO -a $ALPHA -o sim/lasso/data
