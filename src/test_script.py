import yaml
from dsl.models import LLM
from dsl.runner import train_and_eval

import wandb

config = {}
with open("configs/defaults.yaml") as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(base_config)
with open("configs/toxicity/defaults.yaml") as f:
    toxicity_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(toxicity_config)

config.update(
    {
        "model_directory": "/cluster/scratch/ewybitul/models",
        "project": "toxicity-detection-test",
        "learning_rate": 1e-6,
        "end_learning_rate": 0.1,
        "model": "test",
        "logging_period": 1,
        "mixed_precision": None,
        "epochs": 1,
        "beta1": 0.9,
        "beta2": 0.999,
        "optimizer": "adamw",
        "weight_decay": 1e-6,
        # "base_model": "jphme/em_german_leo_mistral",
        "base_model": "jphme/em_german_7b_v01",
        "batch_size": 1,
        "early_stopping_enabled": True,
        "early_stopping_epoch": 2,
        "early_stopping_metric": "validation/loss",
        "early_stopping_threshold": 0.4,
        "warmup_ratio": 0.05,
    }
)

wandb.init(project="toxicity-detection-test", config=config)
model = LLM(wandb.config)
train_and_eval(model, wandb.config)
