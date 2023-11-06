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
        "train_data": "data/processed_comments_train_v3.csv",
        "evaluation_data": "data/processed_comments_evaluation_v3.csv",
        "base_model": "deepset/gbert-base",
        "model": "toxicity-detection-baseline",
        "epochs": 2,
    }
)

wandb.init(project="toxicity-detection", config=config)

match wandb.config["base_model"]:
    case "Hate-speech-CNERG/dehatebert-mono-german_labels=2":
        wandb.config.update(
            {"transform_remove_umlauts": True, "transform_lowercase": True},
            allow_val_change=True,
        )


model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
