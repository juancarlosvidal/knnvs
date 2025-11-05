#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128gb
PYTHON=/mnt/beegfs/home/juan.vidal/miniconda3/envs/knncpu/bin/python3.8
command="$PYTHON $1"
echo $command
$command

