from pathlib import Path
from typing import Optional

import numpy as np
import polars as pr
import torch
import wandb.plot
from torch.utils.data import DataLoader
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    CalibrationError,
    F1Score,
    MeanMetric,
    Metric,
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    Recall,
)

import wandb


def train(
    model: torch.nn.Module,
    config: wandb.Config,
    comments_text: pr.Series,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    assert wandb.run is not None
    wandb.watch(model, log="all", log_freq=1024)

    checkpoint_path = Path(config.model_dir) / "checkpoints" / wandb.run.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Default optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    assert optimizer is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_metrics = _setup_metrics(len(config.class_names), stage="train").to(device)
    model = model.to(device)

    example_ct = 0
    model_best, model_latest = None, None
    best_val_loss = float("inf")

    if config.reweigh_loss:
        # TODO: Replace hand-coded class weights with computed ones
        if len(config.class_names) == 2:
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([0.61230458, 2.726089]), reduction="none"
            )
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(
                weight=torch.tensor(
                    [
                        1.2195804195804196,
                        4.298591549295774,
                        5.482634730538922,
                        3.760164271047228,
                        0.4910041560530902,
                        10.230167597765362,
                        0.9755993606819393,
                        0.2614319366121779,
                        4.36,
                        0.7780752071383047,
                    ]
                ),
                reduction="none",
            )
    else:
        if len(config.class_names) == 2:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_fn = loss_fn.to(device)

    for epoch in range(config.epochs):
        model.train()

        print(f"Epoch {epoch}")
        for step, (_, comments, labels) in enumerate(train_loader):
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(input_ids=comments, labels=labels)
            optimizer.zero_grad()
            loss = loss_fn(outputs.logits, labels)
            loss.sum().backward()
            optimizer.step()

            train_metrics_vals = train_metrics(value=loss)

            example_ct += labels.size(0)
            if step % 512 == 0:
                wandb.log(train_metrics_vals, step=example_ct)

        val_metrics = _evaluate(
            model=model,
            comments_text=comments_text,
            class_names=config.class_names,
            loader=val_loader,
            log_examples=False,
            stage="validation",
        )

        if val_metrics["validation/loss"] < best_val_loss:
            best_val_loss = val_metrics["validation/loss"]
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

        if (
            config.checkpoint_every_n is not None
            and epoch % config.checkpoint_every_n == 0
        ):
            torch.save(
                model_latest,
                checkpoint_path / f"{config.model_name}_epoch={epoch}.ckpt",
            )

        wandb.log(val_metrics, step=example_ct)

    torch.save(model_latest, checkpoint_path / f"{config.model_name}_latest.ckpt")
    torch.save(model_best, checkpoint_path / f"{config.model_name}_best.ckpt")
    if config.log_model_to_wandb:
        model_artifact = wandb.Artifact(
            config.model_name, type="model", metadata=dict(config)
        )
        model_artifact.add_file(checkpoint_path / f"{config.model_name}_best.ckpt")
        wandb.log_artifact(model_artifact, aliases=["best", "latest"])


def evaluate(
    model: torch.nn.Module,
    comments_text: pr.Series,
    config: wandb.Config,
    loader: DataLoader,
):
    metrics = _evaluate(
        model=model,
        comments_text=comments_text,
        loader=loader,
        log_examples=True,
        log_n_worst=config.log_n_worst,
        class_names=config.class_names,
        stage="evaluation",
    )

    wandb.log(metrics)


def _evaluate(
    model: torch.nn.Module,
    comments_text: pr.Series,
    loader: DataLoader,
    log_examples: bool,
    class_names: list[str],
    stage: str,
    log_n_worst: Optional[int] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = _setup_metrics(len(class_names), stage=stage).to(device)
    model.eval()

    with torch.inference_mode():
        comments_batched, logits_batched, labels_batched = [], [], []

        for idx, comments, labels in loader:
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(input_ids=comments, labels=labels)
            logits = outputs.logits
            metrics.update(
                value=outputs.loss.item() * labels.size(0),
                preds=logits,
                target=labels.long(),
            )

            if log_examples:
                comments_batched.append(comments_text[idx.tolist()])
                logits_batched.append(logits.cpu())
                labels_batched.append(labels.cpu())

        if log_examples:
            comments = [comment for batch in comments_batched for comment in batch]
            logits = torch.cat(logits_batched, dim=0)
            labels = torch.cat(labels_batched, dim=0)
            losses = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)

            if log_n_worst is None:
                log_n_worst = losses.size(0)
            _, indices = torch.topk(losses, min(log_n_worst, losses.size(0)))
            predictions = torch.argmax(logits[indices], dim=1)
            _log_sample_predictions(
                [comments[i] for i in indices],
                predictions,
                labels[indices],
                logits.softmax(dim=1),
                class_names,
                stage=stage,
            )

        return _compute_metrics(class_names, metrics)


def _setup_metrics(num_classes: int, stage: str):
    if num_classes == 2:
        kwargs = {"task": "multiclass", "num_classes": 2}
    else:
        kwargs = {"task": "multilabel", "num_labels": num_classes}

    if stage == "train":
        metrics = {"loss": MeanMetric()}
    else:
        metrics: dict[str, Metric] = {
            "f1": F1Score(average="none", **kwargs),
            "auprc": AveragePrecision(average="none", **kwargs),
            "accuracy": Accuracy(average="macro", **kwargs),
            "precision": Precision(average="none", **kwargs),
            "recall": Recall(average="none", **kwargs),
            "pr_curve": PrecisionRecallCurve(thresholds=20, **kwargs),
            "loss": MeanMetric(),
        } | {
            f"f1_{i}": F1Score(average="none", threshold=thr, **kwargs)
            for i, thr in enumerate(np.arange(0, 1.01, 0.05))
        }
        if num_classes == 2:
            metrics["calibration_error"] = CalibrationError(**kwargs)

    return MetricCollection(metrics, prefix=f"{stage}/")


def _log_pr_curve(class_name, precision, recall, thresholds, stage):
    pr_curve_table = wandb.Table(columns=["threshold", "precision", "recall", "class"])
    for t, p, r in zip(thresholds, precision, recall):
        pr_curve_table.add_data(t, p, r, class_name)
    return wandb.plot.line(pr_curve_table, "recall", "precision", title="PR Curve")


def _compute_metrics(class_names, metrics):
    results = metrics.compute()
    metrics.reset()
    f1_curve_tables = {}
    stage = ""
    for name, value in list(results.items()):
        if "pr_curve" in name:
            stage, _ = name.split("/")
            precision, recall, thresholds = value
            for i in range(len(class_names)):
                results[f"{stage}/{class_names[i]}/pr_curve"] = _log_pr_curve(
                    class_names[i], precision[i], recall[i], thresholds, stage
                )
        elif "f1_" in name:
            # HACK: Create a custom Metric for this
            threshold = np.arange(0, 1.01, 0.05)[int(name.split("f1_")[1])]
            for i in range(len(class_names)):
                table = f1_curve_tables.setdefault(
                    class_names[i], wandb.Table(columns=["threshold", "f1", "class"])
                )
                table.add_data(threshold, value[i], class_names[i])
        elif len(value.shape) == 0:
            results[name] = value.item()
        else:
            # Store the per-class values
            stage, bare_name = name.split("/")
            results.update(
                {
                    f"{stage}/{cls}/{bare_name}": val
                    for cls, val in zip(class_names, value)
                }
            )
            # Store a macro-average, too
            results[name] = value.mean().item()
    if f1_curve_tables:
        for i in range(len(class_names)):
            results[f"{stage}/{class_names[i]}/f1_curve"] = wandb.plot.line(
                f1_curve_tables[class_names[i]], "threshold", "f1", title="F1 Curve"
            )

    return results


def _log_sample_predictions(
    comments, predictions, true_labels, probabilities, class_names, stage
):
    table = wandb.Table(columns=["comment", "prediction", "target", *class_names])
    for text, pred, targ, prob in zip(
        comments,
        predictions.to("cpu"),
        true_labels.to("cpu"),
        probabilities.to("cpu"),
    ):
        table.add_data(text, pred, targ, *prob.numpy())
    wandb.log({f"{stage}/predictions_table": table}, commit=False)
