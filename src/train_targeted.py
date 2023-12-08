import yaml
from dsl.models import MultiClassAdapterModule
from dsl.runner import train_and_eval

import wandb

config = {}
with open("configs/defaults.yaml") as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(base_config)
with open("configs/targeted/defaults.yaml") as f:
    toxicity_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(toxicity_config)

config.update(
    {
        "model_directory": "/cluster/scratch/ewybitul/models",
        "early_stopping_enabled": True,
        "early_stopping_epoch": 2,
        "early_stopping_metric": "validation/auprc",
        "early_stopping_threshold": 0.5,
        "hidden_dropout_prob": 0.1,
        "epochs": 5,
        "beta2": 0.999,
    }
)


wandb.init(project="targeted-detection", config=config)

model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
