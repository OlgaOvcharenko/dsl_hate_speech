#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:32G
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --time=01:00:00

CONSUL_HTTP_ADDR=""

mkdir -p logs
module load eth_proxy gcc/11.4.0 python/3.11.6 cuda/12.1.1 
source ".venv_llama/bin/activate"

export WANDB__SERVICE_WAIT=300
export TRANSFORMERS_CACHE=/cluster/scratch/oovcharenko/dsl_hate_speech/cache/

nvidia-smi

python src/scripts/save_models_locally.py


