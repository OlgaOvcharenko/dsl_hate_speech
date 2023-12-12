#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpumem:32G
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

CONSUL_HTTP_ADDR=""

mkdir -p logs
mkdir -p outputs
mkdir -p outputs_targets
mkdir -p data/llm_target

module load eth_proxy gcc/11.4.0 python/3.11.6 cuda/12.1.1 
source ".venv_llama/bin/activate"

export WANDB__SERVICE_WAIT=300
export TRANSFORMERS_CACHE=/cluster/scratch/oovcharenko/dsl_hate_speech/cache/
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

echo "$(date)"
echo "$1"

nvidia-smi

#export CUDA_VISIBLE_DEVICES=0,1

# python -m torch.distributed.launch "$1"
#torchrun --standalone --nnodes=1 "$1"
python "$1" &

sleep 122
nvidia-smi
wait

echo "$(date)"
