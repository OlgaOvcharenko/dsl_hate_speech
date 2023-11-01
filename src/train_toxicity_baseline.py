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
        "model_directory": "/cluster/scratch/ewybitul/models",
        "base_model": "Hate-speech-CNERG/dehatebert-mono-german_labels=2",
        "model": "toxicity-detection-baseline",
        "epochs": 5,
    }
)

wandb.init(project="toxicity-detection", config=config)
model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
