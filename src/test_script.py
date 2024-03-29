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
        "model_directory": "/cluster/scratch/ewybitul/models",
        "model_base": "xlm-roberta-large",
        # "dataset_portion": 0.001,
        "learning_rate": 1e-6,
        "use_learning_rate_finder": True,
        "learning_rate_finder_steps": 500,
        "end_learning_rate": 0.1,
        "model": "test",
        "logging_period": 1,
        "epochs": 2,
    }
)

wandb.init(project="toxicity-detection-test", config=config)
model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
