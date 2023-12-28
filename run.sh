#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=12G
#SBATCH --time=00:10:00

run_toxicity=true

mkdir -p logs
module load eth_proxy gcc/11.4.0 python/3.11.6 cuda/12.1.1 
source ".venv/bin/activate"

export WANDB__SERVICE_WAIT=300

python "$1"

