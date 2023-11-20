import torch

import wandb
from dsl.dataset import (
    class_weights_eff_num,
    class_weights_inverse_frequency,
    setup_datasets,
    setup_loader,
)
from dsl.train import evaluate, train
from dsl.utils import seed_everywhere

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


def train_and_eval(model: torch.nn.modules.module.Module, config: wandb.Config):
    seed_everywhere(config.seed)

    train_df, train_dataset, val_dataset = setup_datasets(config, stage="fit")  # type: ignore
    train_loader = setup_loader(
        train_dataset, shuffle=True, batch_size=config.batch_size
    )
    val_loader = setup_loader(val_dataset, shuffle=False, batch_size=config.batch_size)

    print(f"Starting training with {len(train_dataset)} examples...")

    class_weights = None
    if config.class_weight == "effective_num":
        class_weights = class_weights_eff_num(train_df, config.class_names, config.beta)
    elif config.class_weight == "inverse_frequency":
        class_weights = class_weights_inverse_frequency(train_df, config.class_names)

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
        eval_df, eval_dataset = setup_datasets(config, stage="test")  # type: ignore
        eval_loader = setup_loader(
            eval_dataset, shuffle=False, batch_size=config.batch_size
        )
        print(f"Starting evaluation with {len(eval_df)} examples...")
        evaluate(
            model=best_model,
            comments_text=eval_df["comment"],
            config=config,
            loader=eval_loader,
        )
