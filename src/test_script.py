import yaml

import wandb
from dsl.models import MultiClassAdapterModule
from dsl.runner import train_and_eval

config = {}
with open("configs/defaults.yaml") as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(base_config)
with open("configs/toxicity/defaults.yaml") as f:
    toxicity_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(toxicity_config)

config.update(
    {
        "project": "toxicity-detection-test",
        "dataset_portion": 0.001,
        "model": "test",
        "logging_period": 1,
        "epochs": 1,
    }
)

wandb.init(project="toxicity-detection-test", config=config)
model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
