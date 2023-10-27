import wandb
from dsl.models import MultiLabelModule
from dsl.runner import train_and_eval

optimizer_config = {
    "optimizer": "SGD",
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "weight_decay": 1e-5,
}

training_config = {
    "epochs": 2,
    "batch_size": 16,
    "debug_subset": 200,
    "checkpoint_every_n": 1,
    "log_n_worst": 100,
    "log_model_to_wandb": False,
}

model_config = {
    "model_name": "target-detection-baseline",
    "model_dir": "./models",
    "base_model_id": "Hate-speech-CNERG/dehatebert-mono-german_labels=10",
    "layers_to_freeze": list(range(11)),
}

wandb.init(
    project="target-detection-baseline",
    config={
        "seed": 42,
        "data_path": "./data/clean_comments_non-fr.csv",
        "class_names": [
            "gender",
            "age",
            "sexuality",
            "religion",
            "nationality",
            "disability",
            "social_status",
            "political_views",
            "appearance",
            "other",
        ],
    }
    | optimizer_config
    | training_config
    | model_config,
)

model = MultiLabelModule(wandb.config)
train_and_eval(model, wandb.config)
