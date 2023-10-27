import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('model', nargs='+')
args = parser.parse_args()

wandb.init(
    project="toxicity-detection-baseline",
    config={
        "seed": 42,
        "optimizer": "SGD",
        "learning_rate": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "data_path": "./data/clean_comments_non-fr.csv",
        "model_dir": "./models",
        "base_model_id": args.model if args else "Hate-speech-CNERG/dehatebert-mono-german",
        "logits_path": "results/",
        "model_name": "toxicity-baseline",
        "epochs": 10,
        "batch_size": 16,
        "debug_subset": None,
        "checkpoint_every_n": 3,
        "log_n_worst": 100,
        "log_model_to_wandb": True,
    },
)
config = wandb.config

os.environ["PL_GLOBAL_SEED"] = str(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

comments_text, comments, labels = setup_data(config, debug_subset=config.debug_subset)
train_dataset, val_dataset, test_dataset = setup_datasets(comments, labels)  # type: ignore
train_loader = setup_loader(train_dataset, shuffle=True, batch_size=config.batch_size)
val_loader = setup_loader(val_dataset, shuffle=False, batch_size=config.batch_size)
test_loader = setup_loader(test_dataset, shuffle=False, batch_size=config.batch_size)

model = PretrainedModule(config)

train(
    model=model,
    comments_text=comments_text,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
)

test(model=model, comments_text=comments_text, config=config, loader=test_loader)
