from typing import Any

import torch
import wandb.plot
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


def setup_metrics(num_classes: int, stage: str):
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

    return MetricCollection(metrics, prefix=f"{stage}/")


def setup_macro_f1(stage: str):
    return MetricCollection(
        {"macro_f1": F1Score(average="macro", task="multiclass", num_classes=2)},
        prefix=f"{stage}/",
    )


def process_metrics(metric_values: dict[str, Any], class_names: list[str]):
    results = {}
    for name, value in metric_values.items():
        if "pr_curve" in name:
            results.update(_process_pr_curve(name, value, class_names))
        # elif "f1_" in name:
        #     pass
        elif len(value.shape) == 0:
            results[name] = value.item()
        # elif len(value.shape) == 1 and len(value) == len(class_names):
        #     results.update(_process_per_class_metric(name, value, class_names))

    # results.update(_process_f1_curve(metric_values, class_names))

    return results


def log_sample_predictions(
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


# def _process_per_class_metric(metric_name, value, class_names):
#     result = {}
#     stage, name = metric_name.split("/")
#     if isinstance(value, torch.Tensor) and len(value) == len(class_names):
#         result.update(
#             {
#                 f"{stage}/{cls}/{name}": val.item()
#                 for cls, val in zip(class_names, value)
#             }
#         )
#         result[f"{stage}/macro/{name}"] = value.mean().item()
#     return result


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

    # results = {}
    # for i, class_name in enumerate(class_names):
    #     table = wandb.Table(columns=["threshold", "precision", "recall", "f1", "class"])
    #     for t, p, r in zip(thresholds, precision[i], recall[i]):
    #         table.add_data(t, p, r, class_name)
    #     results[f"{stage}/{class_name}/{name}"] = wandb.plot.line(
    #         table,
    #         "recall",
    #         "precision",
    #         title=f"Precision v. Recall ({class_name})",
    #     )
    # return results


# def _process_f1_curve(metrics: dict[str, Any], class_names: list[str]):
#     tables = {}
#     stage = ""
#     for name, value in metrics.items():
#         if "f1_" in name:
#             stage, num = name.split("/f1_")
#             pos_threshold = np.arange(0, 1.01, 0.01)[int(num)]

#             for val, class_name in zip(value, class_names):
#                 if len(class_names) == 2 and class_name == class_names[0]:
#                     threshold = 1 - pos_threshold
#                 else:
#                     threshold = pos_threshold
#                 table = tables.setdefault(
#                     class_name, wandb.Table(columns=["threshold", "f1", "class"])
#                 )
#                 table.add_data(threshold, val.item(), class_name)

#     results = {}
#     for class_name, table in tables.items():
#         key = f"{stage}/{class_name}/f1_curve"
#         results[key] = wandb.plot.line(
#             tables[class_name],
#             "threshold",
#             "f1",
#             title=f"F1 v. Threshold ({class_name})",
#         )
#     return results
