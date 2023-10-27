import torch

import wandb
from dsl.datasets import setup_datasets, setup_loader
from dsl.train import evaluate, train
from dsl.utils import seed_everywhere


def train_and_eval(model: torch.nn.modules.module.Module, config: wandb.Config):
    seed_everywhere(config.seed)

    train_df, train_dataset, val_dataset = setup_datasets(config, stage="fit")  # type: ignore
    train_loader = setup_loader(
        train_dataset, shuffle=True, batch_size=config.batch_size
    )
    val_loader = setup_loader(val_dataset, shuffle=False, batch_size=config.batch_size)

    train(
        model=model,
        comments_text=train_df["comment"],
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    eval_df, eval_dataset = setup_datasets(config, stage="test")  # type: ignore
    eval_loader = setup_loader(
        eval_dataset, shuffle=False, batch_size=config.batch_size
    )
    evaluate(
        model=model,
        comments_text=eval_df["comment"],
        config=config,
        loader=eval_loader,
    )
