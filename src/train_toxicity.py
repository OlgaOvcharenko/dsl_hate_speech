import yaml
from dsl.models import MultiClassAdapterModule
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
        "model": "toxicity-detection",
        "early_stopping_enabled": True,
        "early_stopping_epoch": 2,
        "early_stopping_metric": "validation/auprc",
        "early_stopping_threshold": 0.6,
        "hidden_dropout_prob": 0.1,
        "epochs": 5,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        # TODO Remove
        "base_model": "microsoft/mdeberta-v3-base",
        "learning_rate": 1e-4,
        "dataset_portion": 0.001,
        "logging_period": 1,
        "create_checkpoints": False,
    }
)

wandb.init(project="toxicity-detection-test", config=config)

model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
