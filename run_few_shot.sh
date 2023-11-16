#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00

mkdir -p logs

module load eth_proxy gcc/8.2.0 python_gpu/3.11.2

nvidia-smi

source ".venv/bin/activate"

python few-shot.py
