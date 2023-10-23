#!/bin/bash

#SBATCH -o logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=64000MB

mkdir -p logs

module load python

source "python_venv/bin/activate"

python3 utils/create_lang_labels.py
