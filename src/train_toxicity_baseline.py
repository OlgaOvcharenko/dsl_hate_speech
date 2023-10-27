import wandb
from dsl.models import MultiClassModule
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
    "debug_subset": 100,
    "checkpoint_every_n": 1,
    "log_n_worst": 100,
    "log_model_to_wandb": False,
}

model_config = {
    "model_name": "toxicity-detection-baseline",
    "model_dir": "./models",
    "base_model_id": "Hate-speech-CNERG/dehatebert-mono-german_labels=2",
    "layers_to_freeze": list(range(11)),
}

data_config = {
    "train_data": "data/processed_comments_train_v1.csv",
    "evaluation_data": "data/processed_comments_evaluation_v1.csv",
    "validation_split": 0.2,
}

wandb.init(
    project="toxicity-detection-baseline",
    config={
        "seed": 42,
        "data_path": "./data/clean_comments_non-fr.csv",
        "class_names": ["non_noxic", "toxic"],
    }
    | data_config
    | optimizer_config
    | training_config
    | model_config,
)

model = MultiClassModule(wandb.config)
train_and_eval(model, wandb.config)
