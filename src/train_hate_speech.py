import yaml
from dsl.models import MultiClassAdapterModule
from dsl.runner import train_and_eval

import wandb

config = {}
with open("configs/defaults.yaml") as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(base_config)
with open("configs/hate_speech/defaults.yaml") as f:
    toxicity_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(toxicity_config)

config.update({"model_directory": "/cluster/scratch/ewybitul/models"})


wandb.init(project="hate-speech-unified", config=config)

model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
