import os
import random

import numpy as np
import torch

import wandb
from baseline import (
    PretrainedModule,
    setup_data,
    setup_datasets,
    setup_loader,
    test,
    train,
)

wandb.init(
    project="toxicity-detection-baseline",
    config={
        "seed": 42,
        "optimizer": "SGD",
        "learning_rate": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "data_path": "./data/all_DEFR_comments_27062022.csv",
        "model_dir": "./models",
        "base_model_id": "Hate-speech-CNERG/dehatebert-mono-german",
        "model_name": "toxicity-baseline",
        "epochs": 5,
        "batch_size": 16,
        "debug_subset": 50,
        "checkpoint_every_n": 3,
        "log_n_worst": 10,
    },
)
config = wandb.config

os.environ["PL_GLOBAL_SEED"] = str(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

data = setup_data(config.data_path, debug_subset=config.debug_subset)
train_dataset, val_dataset, test_dataset = setup_datasets(data)
train_loader = setup_loader(train_dataset, shuffle=True, batch_size=config.batch_size)
val_loader = setup_loader(val_dataset, shuffle=False, batch_size=config.batch_size)
test_loader = setup_loader(test_dataset, shuffle=False, batch_size=config.batch_size)

model = PretrainedModule(config)

train(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
)

test(model=model, config=config, loader=test_loader)