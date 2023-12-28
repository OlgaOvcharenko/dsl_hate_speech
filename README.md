# Hate Speech Detection

To run a hyperparameter sweep:

1. Specify your hyperparameters in a `configs/toxicity/*.yaml` file
2. Check the `SBATCH` requirement tags in `run_agent.sh`. An _agent_ will run one training of some hyperparameter configuration.
3. Run `wandb sweep --project [project name] [path to your .yaml file]` to get a sweep id. It will look like this: `org/entity/id`.
4. Finally, run `bash run_sweep.sh --sweep [sweep id] --count [number of runs]`. This will launch `[number of runs]` jobs with `sbatch`, each of which will run one agent with one hyperparameter combination.