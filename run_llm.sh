#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpumem:24G
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=12G
#SBATCH --time=01:00:00

CONSUL_HTTP_ADDR=""

mkdir -p logs
module load eth_proxy gcc/11.4.0 python/3.11.6 cuda/12.1.1 
source ".venv/bin/activate"

export WANDB__SERVICE_WAIT=300

echo "$(date)"
echo "$1"

nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

# python -m torch.distributed.launch "$1"
torchrun --standalone --nnodes=1 "$1"
#python "$1" &

# sleep 122
# nvidia-smi
# wait

echo "$(date)"
