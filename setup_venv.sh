#!/bin/bash

if [[ ! -d "python_env" ]]; then
  echo "Create Python Virtual Environment on $HOSTNAME"

  # HACK Maybe locally we don't want to use the system site packages
  module load gcc/11.4.0 python/3.11.6 cuda/12.1.1 
  python -m venv .venv_llama
  source ".venv_llama/bin/activate"

  pip install --upgrade pip
  pip install -r requirements.txt
fi
