#!/bin/bash

#SBATCH --output=logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=12G
#SBATCH --time=01:00:00

CONSUL_HTTP_ADDR=""

module load eth_proxy gcc/11.4.0 python/3.11.6 cuda/12.1.1 
source ".venv_llama/bin/activate"

export TRANSFORMERS_CACHE=/cluster/scratch/oovcharenko/dsl_hate_speech/cache/

python src/scripts/save_models_locally.py 

