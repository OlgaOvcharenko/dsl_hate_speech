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
    "epochs": 15,
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
    "model_name": "toxicity-detection-baseline",
    "model_directory": "/cluster/scratch/oovcharenko/models",
    "model": "xml-roberta-large",
    "layers_to_freeze": list(range(11)),
}

data_config = {
    "train_data": "data/processed_comments_train_v1.csv",
    "evaluation_data": "data/processed_comments_evaluation_v1.csv",
    "validation_split": 0.1,
}

wandb.init(
    project="targeted-detection-baseline",
    config={
        "seed": 42,
        "class_names": ["non_targeted", "targeted"],
    }
    | data_config
    | optimizer_config
    | training_config
    | model_config,
)

model = MultiClassModule(wandb.config)
train_and_eval(model, wandb.config)
