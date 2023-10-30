import time
from pathlib import Path
from typing import Callable, Optional

import polars as pr
import torch
import torch.nn.functional as F
import wandb.plot
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

import wandb
from dsl.metrics import (
    log_sample_predictions,
    process_metrics,
    setup_f1_curve,
    setup_metrics,
)


def _get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def _get_optimizer(config: wandb.Config, model: torch.nn.Module):
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")
    return optimizer


def _get_loss(classes_num: int, class_weights: Optional[torch.Tensor] = None):
    if classes_num == 2:
        return torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    else:
        return torch.nn.BCEWithLogitsLoss(weight=class_weights, reduction="none")


def train(
    model: torch.nn.Module,
    config: wandb.Config,
    comments_text: pr.Series,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
):
    assert wandb.run is not None
    # wandb.watch(model, log="all", log_freq=1024)
    device = _get_device()

    checkpoint_path = Path(config.model_directory) / "checkpoints" / wandb.run.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    optimizer = _get_optimizer(config, model)
    train_metrics = setup_metrics(len(config.class_names), stage="train").to(device)
    val_metrics = setup_metrics(len(config.class_names), stage="validation").to(device)
    model = model.to(device)

    example_ct = 0
    model_best, model_latest = None, None
    best_val_loss = float("inf")

    loss_fn = _get_loss(len(config.class_names), class_weights=class_weights)
    loss_fn = loss_fn.to(device)

    start_time = time.time()
    for epoch in range(config.epochs):
        model.train()

        print(f"Epoch {epoch}")
        for step, (_, comments, labels) in enumerate(train_loader):
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(input_ids=comments, labels=labels)
            optimizer.zero_grad()
            loss = loss_fn(outputs.logits, labels)
            loss.mean().backward()
            optimizer.step()

            train_metrics_vals = train_metrics(
                value=loss,
                preds=outputs.logits.softmax(dim=1),
                target=labels.long(),
            )

            example_ct += labels.size(0)
            if (step + 1) % config.logging_period == 0:
                metrics = {"throughput": example_ct / (time.time() - start_time)}
                metrics.update(process_metrics(train_metrics_vals, config.class_names))
                wandb.log(metrics, step=example_ct)

        train_metrics.reset()

        val_metrics_vals = _evaluate(
            model=model,
            comments_text=comments_text,
            class_names=config.class_names,
            metrics=val_metrics,
            loader=val_loader,
            log_examples=False,
            stage="validation",
            loss_fn=loss_fn,
        )
        wandb.log(val_metrics_vals, step=example_ct)

        if val_metrics_vals["validation/loss"] < best_val_loss:
            best_val_loss = val_metrics_vals["validation/loss"]
            model_best = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        model_latest = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        if config.create_checkpoints and (epoch + 1) % config.checkpoint_period == 0:
            torch.save(
                model_latest,
                checkpoint_path / f"{config.model}_epoch={epoch}.ckpt",
            )

    torch.save(model_latest, checkpoint_path / f"{config.model}_latest.ckpt")
    torch.save(model_best, checkpoint_path / f"{config.model}_best.ckpt")
    if config.log_model_to_wandb:
        model_artifact = wandb.Artifact(
            config.model, type="model", metadata=dict(config)
        )
        model_artifact.add_file(checkpoint_path / f"{config.model}_best.ckpt")
        wandb.log_artifact(model_artifact, aliases=["best", "latest"])


def evaluate(
    model: torch.nn.Module,
    comments_text: pr.Series,
    config: wandb.Config,
    loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
):
    device = _get_device()
    metrics = setup_metrics(len(config.class_names), stage="evaluation").to(device)
    metric_vals = _evaluate(
        model=model,
        comments_text=comments_text,
        metrics=metrics,
        loader=loader,
        log_examples="evaluation" in config.log_hardest_examples,
        examples_to_log=config.examples_to_log,
        class_names=config.class_names,
        stage="evaluation",
        loss_fn=_get_loss(len(config.class_names), class_weights),
    )

    wandb.log(metric_vals)


def _evaluate(
    model: torch.nn.Module,
    comments_text: pr.Series,
    metrics: MetricCollection,
    loader: DataLoader,
    log_examples: bool,
    class_names: list[str],
    stage: str,
    loss_fn: Callable,
    examples_to_log: Optional[int] = None,
):
    device = _get_device()
    model.eval()
    f1_curve = setup_f1_curve(num_labels=len(class_names), stage=stage).to(device)

    with torch.inference_mode():
        comments_batched, logits_batched, labels_batched = [], [], []

        for idx, comments, labels in loader:
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(input_ids=comments, labels=labels)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            metrics.update(
                value=loss, preds=logits.softmax(dim=1), target=labels.long()
            )
            f1_curve.update(
                preds=logits.softmax(dim=1),
                target=labels.long()
                if len(class_names) > 2
                else F.one_hot(labels, num_classes=2),
            )

            if log_examples:
                comments_batched.append(comments_text[idx.tolist()])
                logits_batched.append(logits.cpu())
                labels_batched.append(labels.cpu())

        if log_examples:
            comments = [comment for batch in comments_batched for comment in batch]
            logits = torch.cat(logits_batched, dim=0)
            labels = torch.cat(labels_batched, dim=0)
            losses = loss_fn(logits, labels)
            if len(class_names) > 2:
                losses = losses.median(dim=1).values

            if examples_to_log is None:
                examples_to_log = losses.size(0)
            _, indices = torch.topk(
                losses, min(examples_to_log, losses.size(0))  # type: ignore
            )
            predictions = torch.argmax(logits[indices], dim=1)
            log_sample_predictions(
                comments=[comments[i] for i in indices],
                predictions=predictions,
                true_labels=labels[indices],
                probabilities=logits.softmax(dim=1),
                class_names=class_names,
                stage=stage,
            )

        metric_values = metrics.compute()
        metric_values.update(f1_curve.compute())
        f1_curve.reset()
        metrics.reset()
        return process_metrics(metric_values, class_names)
