import wandb
from dsl.models import MultiClassModule
from dsl.runner import train_and_eval

user = "oovcharenko" if True else "ewybitul"

optimizer_config = {
    "optimizer": "SGD",
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "weight_decay": 1e-5,
}

training_config = {
    "epochs": 30,
    "batch_size": 16,
    "debug_subset": None,
    "log_every_nth_step": 512,
    "checkpoint_every_nth_epoch": 1,
    "log_n_worst": 100,
    "log_model_to_wandb": True,
    "reweigh_loss": "effective_num",
    "beta": 0.999,
}

model_config = {
    "model_name": "toxicity-detection-baseline",
    "model_dir": f"/cluster/scratch/{user}/dsl_hate_speech/models",
    "base_model_id": "Hate-speech-CNERG/dehatebert-mono-german",
    "layers_to_freeze": list(range(9)),
}

data_config = {
    "train_data": "data/processed_comments_train_v3.csv",
    "evaluation_data": "data/processed_comments_evaluation_v3.csv",
    "validation_split": 0.1,
}

wandb.init(
    project="target-group-detection",
    config={
        "seed": 42,
        "class_names": ["non_political_views", "political_views"],
    }
    | data_config
    | optimizer_config
    | training_config
    | model_config,
)

model = MultiClassModule(wandb.config)
train_and_eval(model, wandb.config)
