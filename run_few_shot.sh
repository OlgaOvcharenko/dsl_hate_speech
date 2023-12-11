#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

mkdir -p logs

module load gcc/11.4.0 python/3.11.6 cuda/12.1.1 

nvidia-smi

source ".venv/bin/activate"

python few-shot.py
