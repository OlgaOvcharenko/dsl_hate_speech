#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:32G
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=12G
#SBATCH --time=01:00:00

CONSUL_HTTP_ADDR=""

mkdir -p logs
module load eth_proxy gcc/8.2.0 python_gpu/3.11.2
source ".venv/bin/activate"

export WANDB__SERVICE_WAIT=300

echo "$(date)"
echo "$1"

nvidia-smi

# python -m torch.distributed.launch "$1"
torchrun --nproc_per_node 2 "$1"

echo "$(date)"
