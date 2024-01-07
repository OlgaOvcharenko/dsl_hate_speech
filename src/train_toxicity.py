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
        "seed": 42,
        "task": "toxicity",
        "beta1": 0.9,
        "beta2": 0.999,
        "epochs": 5,
        "optimizer": "adamw",
        "base_model": "xlm-roberta-large",
        "batch_size": 16,
        "train_data": "data/processed_training_main_v4.csv",
        "class_names": ["non_toxic", "toxic"],
        "adapter_type": "default",
        "class_weight": "unchanged",
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
        "learning_rate": 0.0001,
        "logging_period": 1024,
        "dataset_portion": 1,
        "evaluation_data": [
            "data/processed_evaluation_expert_v4.csv",
            "data/processed_evaluation_representative_v4.csv",
            "data/processed_evaluation_main_v4.csv",
        ],
        "evaluation_data_names": ["expert", "representative", "main"],
        "examples_to_log": 500,
        "mixed_precision": None,
        "model_directory": "/cluster/scratch/ewybitul/models",
        "transform_clean": True,
        "transform_lowercase": False,
        "transform_remove_umlauts": False,
        "validation_split": 0.1,
        "checkpoint_period": 1,
        "create_checkpoints": True,
        "log_model_to_wandb": False,
        "hidden_dropout_prob": 0.1,
        "early_stopping_epoch": 2,
        "log_hardest_examples": ["evaluation"],
        "early_stopping_metric": "validation/auprc",
        "early_stopping_enabled": True,
        "early_stopping_threshold": 0.6,
        "use_learning_rate_finder": False,
    }
)

wandb.init(project="toxicity-detection", config=config)

model = MultiClassAdapterModule(wandb.config)
train_and_eval(model, wandb.config)
