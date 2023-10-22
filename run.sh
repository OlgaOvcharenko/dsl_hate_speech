#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

HF_DATASETS_OFFLINE=1 
TRANSFORMERS_OFFLINE=1

mkdir -p logs
mkdir -p models_saved

module load eth_proxy gcc/8.2.0 python_gpu/3.11.2

nvidia-smi
echo

source ".venv/bin/activate"

export WANDB__SERVICE_WAIT=300
python test_baseline.py