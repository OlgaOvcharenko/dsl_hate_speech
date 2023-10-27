#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:15g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00

mkdir -p logs

module load eth_proxy gcc/8.2.0 python_gpu/3.11.2

nvidia-smi
echo

source ".venv/bin/activate"

export WANDB__SERVICE_WAIT=300
python src/train_toxicity_baseline.py
# python src/train_target_group_baseline.py