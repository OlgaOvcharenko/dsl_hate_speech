# Hate Speech Detection

The rapid development of online media has given individuals a platform for expression and debate from the comfort of their homes. Yet, this freedom has inadvertently fueled a surge in online toxicity and hate speech. Social networks, forums, and chat room moderators are overwhelmed, as seen with Twitter, where users posted 500 million tweets per day in 2016. Current automatic detection methods fall short, particularly in pinpointing the targets of hate speech. The Hate Speech Detection Project seeks to address this gap, providing a robust hate speech classification and target group detection models, which can be built upon to help Swiss and German newspaper moderators to combat hate speech, and to extract deeper insights from online data for research purposes.

All run scripts use SLURM workload manager and WandB.


To run a toxicity/hate speech detection:

1. `sbatch run.sh src/[path to file].py`, e.g., `sbatch run.sh src/train_toxicity_roberta.py`


To run a target group detection:

1. LLM: Llama 2 7b:  `sbatch run_llm_2g.sh src/dsl/llm_targets.py`. There are also scripts to run Mistral 7b in `src/dsl/llm_targets.py`
2. Few-shot with TARS classifier: `sbatch run_llm_2g.sh src/few-shot.py`


To run a hyperparameter sweep:

1. Specify your hyperparameters in a `configs/toxicity/*.yaml` file
2. Check the `SBATCH` requirement tags in `run_agent.sh`. An _agent_ will run one training of some hyperparameter configuration.
3. Run `wandb sweep --project [project name] [path to your .yaml file]` to get a sweep id. It will look like this: `org/entity/id`.
4. Finally, run `bash run_sweep.sh --sweep [sweep id] --count [number of runs]`. This will launch `[number of runs]` jobs with `sbatch`, each of which will run one agent with one hyperparameter combination.
