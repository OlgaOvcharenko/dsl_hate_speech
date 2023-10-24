import torch

import wandb
from dsl.datasets import load_data, setup_datasets, setup_loader
from dsl.train import test, train
from dsl.utils import seed_everywhere


def train_and_eval(model: torch.nn.modules.module.Module, config: wandb.Config):
    seed_everywhere(config.seed)

    df, comments, labels = load_data(config, debug_subset=config.debug_subset)
    train_dataset, val_dataset, test_dataset = setup_datasets(comments, labels, config)
    train_loader = setup_loader(
        train_dataset, shuffle=True, batch_size=config.batch_size
    )
    val_loader = setup_loader(val_dataset, shuffle=False, batch_size=config.batch_size)
    test_loader = setup_loader(
        test_dataset, shuffle=False, batch_size=config.batch_size
    )

    train(
        model=model,
        comments_text=df["comment"],
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    test(
        model=model,
        comments_text=df["comment"],
        config=config,
        loader=test_loader,
    )
