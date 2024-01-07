import yaml
from dsl.models import MultiClassAdapterModule
from dsl.runner import train_and_eval

import wandb

config = {}
with open("configs/defaults.yaml") as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(base_config)
with open("configs/hate_speech_unified/defaults.yaml") as f:
    hate_speech_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(hate_speech_config)

config.update(
    {
        "base_model": "xlm-roberta-large",
        "model": "hate-speech-unified",
        "epochs": 5,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "batch_size": 16,
        "hidden_dropout_prob": 0.15,
        # TODO Remove
        # "logging_period": 1,
        # "dataset_portion": 0.001,
        # "learning_rate": 1e-4,
        # "train_set_balance": 0.5,
        # TODO Add
        "early_stopping_enabled": True,
        "early_stopping_epoch": 2,
        "early_stopping_metric": "validation/auprc",
        "early_stopping_threshold": 0.58,
    }
)


wandb.init(project="hate-speech-unified", config=config)

model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
