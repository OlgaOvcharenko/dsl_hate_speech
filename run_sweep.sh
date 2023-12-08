#!/bin/bash

module load eth_proxy gcc/8.2.0 python_gpu/3.11.2
source ".venv/bin/activate"

# The following code saves the option --count to a variable count using getopts

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    --count)
        count="$2"
        shift
        shift
        ;;
    --sweep)
        sweep="$2"
        shift
        shift
        ;;
    *)
        echo "Usage: $0 --count number --sweep sweep_id"
        exit 1
        ;;
    esac
done

count=$((${count:-1}))

echo "Preparing $count agent(s) to run the sweep "
for ((i = 1; i <= count; i++)); do
    echo "Starting agent $i..."
    sbatch -A s_stud_infk run_agent.sh "$sweep"
done
