from pathlib import Path

import polars as pr
import torch
import wandb.plot
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, Precision, Recall

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

    model.to(device)

    example_ct = 0
    model_best, model_latest = None, None
    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()

        print(f"Epoch {epoch}")
        for step, (_, comments, labels) in enumerate(train_loader):
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(input_ids=comments, labels=labels)
            logits = outputs.logits
            train_loss = outputs.loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(comments)
            if step % 512 == 0:
                wandb.log({"train/loss": train_loss}, step=example_ct)

        val_loss, logits, labels = _evaluate(
            model=model,
            comments_text=comments_text,
            class_names=config.class_names,
            loader=val_loader,
            log_examples=(epoch % config.checkpoint_every_n == 0),
            log_n_worst=config.log_n_worst,
            stage="validation",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

        metrics = _compute_metrics(
            logits, labels, config.class_names, stage="validation"
        )
        metrics.update({"validation/loss": val_loss, "epoch": epoch})

        if len(config.class_names) == 2:
            metrics["validation/pr_curve"] = wandb.plot.pr_curve(
                y_true=labels,
                y_probas=torch.softmax(logits, dim=1),
                labels=config.class_names,
            )

        wandb.log(metrics, step=example_ct)

    torch.save(model_latest, checkpoint_path / f"{config.model_name}_latest.ckpt")
    torch.save(model_best, checkpoint_path / f"{config.model_name}_best.ckpt")
    if config.log_model_to_wandb:
        model_artifact = wandb.Artifact(
            config.model_name, type="model", metadata=dict(config)
        )
        model_artifact.add_file(checkpoint_path / f"{config.model_name}_best.ckpt")
        wandb.log_artifact(model_artifact, aliases=["best", "latest"])


def test(
    model: torch.nn.Module,
    comments_text: pr.Series,
    config: wandb.Config,
    loader: DataLoader,
):
    loss, logits, labels = _evaluate(
        model=model,
        comments_text=comments_text,
        loader=loader,
        log_examples=True,
        log_n_worst=config.log_n_worst,
        class_names=config.class_names,
        stage="test",
    )

    metrics = _compute_metrics(logits, labels, config.class_names, stage="test")
    wandb.summary.update(metrics | {"test/loss": loss})


def _evaluate(
    model: torch.nn.Module,
    comments_text: pr.Series,
    loader: DataLoader,
    log_examples: bool,
    log_n_worst: int,
    class_names: list[str],
    stage: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.inference_mode():
        loss = 0
        comments_batched, logits_batched, labels_batched = [], [], []

        for idx, comments, labels in loader:
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(input_ids=comments, labels=labels)
            logits = outputs.logits

            loss += outputs.loss.item() * labels.size(0)

            if log_examples:
                comments_batched.append(comments_text[idx.tolist()])
            logits_batched.append(logits.cpu())
            labels_batched.append(labels.cpu())

        comments = [comment for batch in comments_batched for comment in batch]
        logits = torch.cat(logits_batched, dim=0)
        labels = torch.cat(labels_batched, dim=0)

        if log_examples:
            losses = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
            _, top_n_indices = torch.topk(losses, min(log_n_worst, losses.size(0)))
            predictions = torch.argmax(logits[top_n_indices], dim=1)
            _log_sample_predictions(
                [comments[i] for i in top_n_indices],
                predictions,
                labels[top_n_indices],
                logits.softmax(dim=1),
                class_names,
                stage=stage,
            )

        total = len(loader.dataset)  # type: ignore
        return loss / total, logits, labels


def _compute_metrics(logits, labels, class_names, stage=""):
    labels = labels.long()
    if len(class_names) == 2:
        task = "multiclass"
        f1_fn = F1Score(task=task, num_classes=2, average="macro")
        auprc_fn = AveragePrecision(task=task, num_classes=2, average="macro")
        auroc_fn = AUROC(task=task, num_classes=2, average="macro")
        accuracy_fn = Accuracy(task=task, num_classes=2, average="macro")
        precision_fn = Precision(task=task, num_classes=2, average="none")
        recall_fn = Recall(task=task, num_classes=2, average="none")
    else:
        n = len(class_names)
        task = "multilabel"
        f1_fn = F1Score(task=task, num_labels=n, average="macro")
        auprc_fn = AveragePrecision(task=task, num_labels=n, average="macro")
        auroc_fn = AUROC(task=task, num_labels=n, average="macro")
        accuracy_fn = Accuracy(task=task, num_labels=n, average="macro")
        precision_fn = Precision(task=task, num_labels=n, average="none")
        recall_fn = Recall(task=task, num_labels=n, average="none")

    f1 = f1_fn(logits, labels).item()
    auprc = auprc_fn(logits, labels).item()
    auroc = auroc_fn(logits, labels).item()
    accuracy = accuracy_fn(logits, labels).item()
    precisions = precision_fn(logits, labels)
    recalls = recall_fn(logits, labels)

    return (
        {
            f"{stage}/f1": f1,
            f"{stage}/auprc": auprc,
            f"{stage}/auroc": auroc,
            f"{stage}/accuracy": accuracy,
        }
        | {f"{stage}/precision_{c}": p for c, p in zip(class_names, precisions)}
        | {f"{stage}/recall_{c}": p for c, p in zip(class_names, recalls)}
    )


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
