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
    "epochs": 10,
    "batch_size": 16,
    "debug_subset": None,
    "log_every_nth_step": 512,
    "checkpoint_every_nth_epoch": 1,
    "log_n_worst": 100,
    "log_model_to_wandb": True,
    "reweigh_loss": True,
}

model_config = {
    "model_name": "target-group-detection-baseline",
    "model_dir": "/cluster/scratch/ewybitul/models",
    "base_model_id": "Hate-speech-CNERG/dehatebert-mono-german_labels=10",
    "layers_to_freeze": list(range(11)),
}

data_config = {
    "train_data": "data/processed_comments_train_v1.csv",
    "evaluation_data": "data/processed_comments_evaluation_v1.csv",
    "validation_split": 0.2,
}

wandb.init(
    project="target-group-detection",
    config={
        "seed": 42,
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
    | data_config
    | optimizer_config
    | training_config
    | model_config,
)

model = MultiLabelModule(wandb.config)
train_and_eval(model, wandb.config)
