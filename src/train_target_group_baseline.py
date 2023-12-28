import wandb
from dsl.models import MultiLabelModule
from dsl.runner import train_and_eval

user = "oovcharenko" if True else "ewybitul"

optimizer_config = {
    "optimizer": "SGD",
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "weight_decay": 1e-5,
}

training_config = {
    "epochs": 10,
    "batch_size": 16,
    "dataset_portion": None,
    "logging_period": 512,
    "checkpoint_period": 1,
    "examples_to_log": 100,
    "log_model_to_wandb": True,
    "class_weight": "effective_num",
    "beta": 0.999,
}

model_config = {
    "model_name": "target-group-detection-baseline",
    "model_dir": f"/cluster/scratch/{user}/models",
    "base_model_id": "Hate-speech-CNERG/dehatebert-mono-german_labels=10",
    "layers_to_freeze": list(range(11)),
}

data_config = {
    "train_data": "data/processed_comments_train_v1.csv",
    "evaluation_data": "data/processed_comments_evaluation_v1.csv",
    "validation_split": 0.1,
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
