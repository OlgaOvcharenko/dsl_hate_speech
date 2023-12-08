import torch

import wandb
from dsl.datasets import (
    class_weights_eff_num,
    class_weights_inverse_frequency,
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
    val_loader = setup_loader(
        val_dataset, shuffle=False, batch_size=config.batch_size
    )

    print(f"Starting training with {len(train_dataset)} examples...")

    class_weights = None
    if config.class_weight == "effective_num":
        class_weights = class_weights_eff_num(
            train_df, config.class_names, config.beta
        )
    elif config.class_weight == "inverse_frequency":
        class_weights = class_weights_inverse_frequency(
            train_df, config.class_names
        )

    if class_weights is not None:
        print("Class weights:")
        for c, w in zip(config.class_names, class_weights):  # type: ignore
            print(f"\t{c}: {w}")
        print()

    best_model = train(
        model=model,
        comments_text=train_df["comment"],
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
    )

    if not config.use_learning_rate_finder:
        names = config.evaluation_data_names
        for name, (eval_df, eval_dataset) in zip(
            names, setup_datasets(config, stage="test")
        ):
            prefix = f"evaluation/{name}"
            eval_loader = setup_loader(
                eval_dataset, shuffle=False, batch_size=config.batch_size
            )
            print(f"Starting evaluation with {len(eval_df)} examples...")
            evaluate(
                model=best_model,
                comments_text=eval_df["comment"],
                config=config,
                prefix=prefix,
                loader=eval_loader,
            )
