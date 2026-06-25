#!/bin/bash
##SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32gb
#SBATCH --job-name=knnvs 
PYTHON=$(which python3)
command="$PYTHON $1"
echo $command
$command

