import torch

import wandb
from dsl.datasets import (
    class_weights_eff_num,
    class_weights_inverse_ratio,
    setup_datasets,
    setup_loader,
)
from dsl.train import evaluate, train
from dsl.utils import seed_everywhere


def train_and_eval(model: torch.nn.modules.module.Module, config: wandb.Config):
    seed_everywhere(config.seed)

    train_df, train_dataset, val_dataset = setup_datasets(config, stage="fit")  # type: ignore
    train_loader = setup_loader(
        train_dataset, shuffle=True, batch_size=config.batch_size
    )
    val_loader = setup_loader(val_dataset, shuffle=False, batch_size=config.batch_size)

    class_weights = None
    if config.reweigh_loss is not None:
        if config.reweigh_loss == "effective_num":
            class_weights = class_weights_eff_num(train_df, config.class_names)
        elif config.reweigh_loss == "inverse_ratio":
            class_weights = class_weights_inverse_ratio(train_df, config.class_names)

        print("Commencing training with class weights:")
        for c, w in zip(config.class_names, class_weights):  # type: ignore
            print(f"\t{c}: {w}")
        print()

    train(
        model=model,
        comments_text=train_df["comment"],
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
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