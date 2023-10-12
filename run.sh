#!/bin/bash

#SBATCH -o logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

mkdir -p logs
mkdir -p models

module load python

nvidia-smi

poetry install
poetry run python ./baseline_test.py
