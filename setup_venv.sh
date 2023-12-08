#!/bin/bash

if [[ ! -d "python_env" ]]; then
  echo "Create Python Virtual Environment on $HOSTNAME"

  # HACK Maybe locally we don't want to use the system site packages
  module load gcc/8.2.0 python_gpu/3.11.2
  python -m venv .venv --system-site-packages
  source ".venv/bin/activate"

  pip install --upgrade pip
  pip install -r requirements.txt
fi
