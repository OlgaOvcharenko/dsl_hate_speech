from typing import Any

import torch
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
import wandb.plot


def setup_metrics(num_classes: int, stage: str, prefix: str | None = None):
    if prefix is None:
        prefix = stage
    if num_classes == 2:
        kwargs = {"task": "binary", "num_classes": 2}
    else:
        kwargs = {"task": "multilabel", "num_labels": num_classes}

    if stage == "train":
        metrics = {
            "loss": MeanMetric(),
            "f1": F1Score(average="macro", **kwargs),
            "auprc": AveragePrecision(average="macro", **kwargs),
            "precision": Precision(average="macro", **kwargs),
            "recall": Recall(average="macro", **kwargs),
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
        }
        if num_classes == 2:
            metrics["calibration_error"] = CalibrationError(**kwargs)

    return MetricCollection(metrics, prefix=f"{prefix}/")


def setup_macro_f1(prefix: str):
    return MetricCollection(
        {
            "macro_f1": F1Score(
                average="macro", task="multiclass", num_classes=2
            ),
            "weighted_f1": F1Score(
                average="weighted", task="multiclass", num_classes=2
            ),
        },
        prefix=f"{prefix}/",
    )


def process_metrics(metric_values: dict[str, Any], class_names: list[str]):
    results = {}
    for name, value in metric_values.items():
        if "pr_curve" in name:
            results.update(_process_pr_curve(name, value, class_names))
        elif len(value.shape) == 0:
            results[name] = value.item()

    return results


def log_sample_predictions(
    comments, predictions, true_labels, probabilities, class_names, prefix
):
    table = wandb.Table(
        columns=["comment", "prediction", "target", *class_names]
    )
    for text, pred, targ, prob in zip(
        comments,
        predictions.to("cpu"),
        true_labels.to("cpu"),
        probabilities.to("cpu"),
    ):
        table.add_data(text, pred, targ, *prob.numpy())
    wandb.log({f"{prefix}/predictions_table": table}, commit=False)


def _process_pr_curve(metric_name: str, value: Any, class_names: list[str]):
    stage, _ = metric_name.split("/")
    precision, recall, thresholds = value

    if len(class_names) == 2:
        table = wandb.Table(columns=["threshold", "precision", "recall", "f1"])
        for t, p, r in zip(thresholds, precision, recall):
            f1 = 2 * (p * r) / (p + r + 1e-20)
            table.add_data(t.item(), p.item(), r.item(), f1.item())
        return {f"{stage}/pr_table": table}
    return {}
