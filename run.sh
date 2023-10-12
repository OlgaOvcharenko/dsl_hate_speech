#!/bin/bash

#SBATCH -o logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

HF_DATASETS_OFFLINE=1 
TRANSFORMERS_OFFLINE=1

mkdir -p logs
mkdir -p models_saved

module load python3.11.*

nvidia-smi

source "python_venv/bin/activate"
pip install --force-reinstall torch==1.10.2+cu113 --extra-index-url https://download.pytorch.org/whl/
python test_baseline.py
