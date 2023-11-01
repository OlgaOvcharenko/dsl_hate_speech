#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2:30:00

mkdir -p logs
# module load eth_proxy gcc/8.2.0 python_gpu/3.11.2
source ".venv/bin/activate"

export WANDB__SERVICE_WAIT=300
wandb agent --count 1 "$1"
