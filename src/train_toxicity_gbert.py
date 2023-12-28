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
        "model_directory": "/cluster/scratch/ewybitul/models",
        "train_data": "data/processed_comments_train_v3.csv",
        "evaluation_data": "data/processed_comments_evaluation_v3.csv",
        "base_model": "deepset/gbert-large",
        "model": "toxicity-detection-baseline",
        "learning_rate": 1e-4,
        "optimizer": "adam",
        "mixed_precision": None,
        "batch_size": 16,
        "epochs": 5,
    }
)

wandb.init(project="toxicity-detection", config=config)

model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
