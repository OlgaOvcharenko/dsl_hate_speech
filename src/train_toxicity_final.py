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
        # Full final training
        "train_data": "data/processed_full_main_v4.csv",
        "evaluation_data": [
            "data/processed_evaluation_expert_v4.csv",
            "data/processed_evaluation_representative_v4.csv",
        ],
        "evaluation_data_names": ["expert", "representative"],
        "validation_split": 0.0,
        # Config for the best toxicity model
        "base_model": "xlm-roberta-large",
        "model": "toxicity-final",
        "hidden_dropout_prob": 0.1,
        "epochs": 5,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "batch_size": 16,
        "learning_rate": 1e-4,
    }
)

wandb.init(project="toxicity-detection", config=config)

model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
