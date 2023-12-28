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
mkdir -p few_shot
mkdir -p few_shot/res

module load eth_proxy gcc/8.2.0 python_gpu/3.11.2

source ".venv/bin/activate"
which python
python few-shot.py
