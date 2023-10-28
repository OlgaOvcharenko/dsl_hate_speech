from pathlib import Path
from typing import Any, Optional

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
    val_metrics = _setup_metrics(len(config.class_names), stage="validation").to(device)
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
            loss.mean().backward()
            optimizer.step()

            preds = outputs.logits.softmax(dim=1)
            train_metrics_vals = train_metrics(
                value=loss,
                preds=preds[:, 1] if len(config.class_names) == 2 else preds,
                target=labels,
            )

            example_ct += labels.size(0)
            if step % config.log_every_nth_step == 0:
                wandb.log(
                    _process_metrics(train_metrics_vals, config.class_names),
                    step=example_ct,
                )

        train_metrics.reset()

        val_metrics_vals = _evaluate(
            model=model,
            comments_text=comments_text,
            class_names=config.class_names,
            metrics=val_metrics,
            loader=val_loader,
            log_examples=False,
            stage="validation",
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

        if (
            config.checkpoint_every_nth_epoch is not None
            and epoch % config.checkpoint_every_nth_epoch == 0
        ):
            torch.save(
                model_latest,
                checkpoint_path / f"{config.model_name}_epoch={epoch}.ckpt",
            )

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = _setup_metrics(len(config.class_names), stage="evaluation").to(device)
    metric_vals = _evaluate(
        model=model,
        comments_text=comments_text,
        metrics=metrics,
        loader=loader,
        log_examples=True,
        log_n_worst=config.log_n_worst,
        class_names=config.class_names,
        stage="evaluation",
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
    log_n_worst: Optional[int] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.inference_mode():
        comments_batched, logits_batched, labels_batched = [], [], []

        for idx, comments, labels in loader:
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(input_ids=comments, labels=labels)
            logits = outputs.logits
            preds = logits.softmax(dim=1)
            metrics.update(
                value=outputs.loss.item() * labels.size(0),
                preds=preds[:, 1] if len(class_names) == 2 else preds,
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

        metric_values = metrics.compute()
        metrics.reset()
        return _process_metrics(metric_values, class_names)


def _process_per_class_metric(metric_name, value, class_names):
    result = {}
    if isinstance(value, torch.Tensor) and len(value) == len(class_names):
        stage, name = metric_name.split("/")
        result.update(
            {
                f"{stage}/{cls}/{name}": val.item()
                for cls, val in zip(class_names, value)
            }
        )
        result[metric_name] = value.mean().item()
    return result


def _setup_metrics(num_classes: int, stage: str):
    if num_classes == 2:
        kwargs = {"task": "binary", "num_classes": 2}
    else:
        kwargs = {"task": "multilabel", "num_labels": num_classes}

    if stage == "train":
        metrics = {
            "loss": MeanMetric(),
            "f1": F1Score(average="none", **kwargs),
            "precision": Precision(average="none", **kwargs),
            "recall": Recall(average="none", **kwargs),
        }
    else:
        metrics: dict[str, Metric] = {
            "f1": F1Score(average="none", **kwargs),
            "auprc": AveragePrecision(average="none", **kwargs),
            "accuracy": Accuracy(average="macro", **kwargs),
            "precision": Precision(average="none", **kwargs),
            "recall": Recall(average="none", **kwargs),
            "pr_curve": PrecisionRecallCurve(
                thresholds=torch.arange(0, 1, 0.01), **kwargs
            ),
            "loss": MeanMetric(),
        } | {
            f"f1_{i}": F1Score(average="none", threshold=thr, **kwargs)
            for i, thr in enumerate(np.arange(0, 1.01, 0.01))
        }
        if num_classes == 2:
            metrics["calibration_error"] = CalibrationError(**kwargs)

    return MetricCollection(metrics, prefix=f"{stage}/")


def _process_pr_curve(metric_name: str, value: Any, class_names: list[str]):
    stage, name = metric_name.split("/")
    precision, recall, thresholds = value
    if len(class_names) == 2:
        table = wandb.Table(columns=["threshold", "precision", "recall"])
        for t, p, r in zip(thresholds, precision, recall):
            table.add_data(t, p, r)
        return {
            f"{stage}/{name}": wandb.plot.line(
                table, "recall", "precision", title="Precision v. Recall"
            )
        }
    else:
        results = {}
        for i, class_name in enumerate(class_names):
            table = wandb.Table(columns=["threshold", "precision", "recall", "class"])
            for t, p, r in zip(thresholds, precision[i], recall[i]):
                table.add_data(t, p, r, class_name)
            results[f"{stage}/{class_name}/{name}"] = wandb.plot.line(
                table,
                "recall",
                "precision",
                title=f"Precision v. Recall ({class_name})",
            )
        return results


def _process_f1_curve(metrics: dict[str, Any], class_names: list[str]):
    if len(class_names) == 2:
        table = wandb.Table(columns=["threshold", "f1_value"])
        stage = ""
        log_table = False
        for name, value in metrics.items():
            if "f1_" in name:
                log_table = True
                stage, num = name.split("/f1_")
                threshold = np.arange(0, 1.01, 0.01)[int(num)]
                table.add_data(threshold, value.item())
        if log_table:
            return {
                f"{stage}/f1_v_threshold": wandb.plot.line(
                    table, "threshold", "f1_value", title=f"F1 v. Threshold"
                )
            }
        else:
            return {}
    else:
        tables = {}
        stage = ""
        for name, value in metrics.items():
            if "f1_" in name:
                stage, num = name.split("/f1_")
                threshold = np.arange(0, 1.01, 0.01)[int(num)]
                for val, class_name in zip(value, class_names):
                    table = tables.setdefault(
                        class_name, wandb.Table(columns=["threshold", "f1", "class"])
                    )
                    table.add_data(threshold, val.item(), class_name)

        results = {}
        for class_name, table in tables.items():
            key = f"{stage}/{class_name}/f1_curve"
            results[key] = wandb.plot.line(
                tables[class_name],
                "threshold",
                "f1",
                title=f"F1 v. Threshold ({class_name})",
            )
        return results


def _process_metrics(metric_values: dict[str, Any], class_names: list[str]):
    results = {}
    for name, value in metric_values.items():
        if "pr_curve" in name:
            results.update(_process_pr_curve(name, value, class_names))
        elif "f1_" in name:
            pass
        elif len(value.shape) == 0:
            results[name] = value.item()
        elif len(value.shape) == 1 and len(value) == len(class_names):
            results.update(_process_per_class_metric(name, value, class_names))

    results.update(_process_f1_curve(metric_values, class_names))

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
