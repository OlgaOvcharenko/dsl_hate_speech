from pathlib import Path
from typing import Callable, Optional

import polars as pr
import numpy as np
import torch
import torchmetrics
import wandb.plot
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb


class CommentDataset(Dataset):
    def __init__(self, comments: torch.Tensor, labels: torch.Tensor):
        self.comments = comments
        self.labels = labels

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        return (index, self.comments[index], self.labels[index])


def setup_data(config: wandb.Config, debug_subset: Optional[int] = None):
    df = pr.read_csv(config.data_path)
    if debug_subset is not None:
        df = df.head(debug_subset)

    name = Path(config.data_path).stem
    tokenized_path = Path("data") / f"{name}_tokenized.pt"
    if tokenized_path.exists():
        tokenized_comments = torch.load(tokenized_path)
    else:
        tokenizer = get_tokenizer(config.model_dir, config.base_model_id)
        tokenized_comments = tokenizer(
            df["comment"].to_list(), padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]
        torch.save(tokenized_comments, tokenized_path)

    labels = torch.tensor(df["toxic"].to_numpy())
    return df["comment"], tokenized_comments, labels


def setup_datasets(
    comments: torch.Tensor,
    labels: torch.Tensor,
    test_split: float = 0.2,
    val_split: float = 0.2,
):
    train_data, test_data, train_labels, test_labels = train_test_split(
        comments.numpy(),
        labels.numpy(),
        test_size=test_split,
        shuffle=True,
        stratify=labels.numpy(),
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data,
        train_labels,
        test_size=val_split,
        shuffle=True,
        stratify=train_labels,
    )

    train_dataset = CommentDataset(torch.tensor(train_data), torch.tensor(train_labels))
    val_dataset = CommentDataset(torch.tensor(val_data), torch.tensor(val_labels))
    test_dataset = CommentDataset(torch.tensor(test_data), torch.tensor(test_labels))

    return train_dataset, val_dataset, test_dataset


def setup_loader(data: Dataset, shuffle: bool, batch_size: int = 16):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


class PretrainedModule(torch.nn.Module):
    def __init__(self, config: wandb.Config):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            f"{config.model_dir}/{config.base_model_id}_model",
            num_labels=2,
            local_files_only=True,
        )

        for name, param in self.model.named_parameters():
            for i in range(11):
                if f"encoder.layer.{i}." in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)


def get_tokenizer(model_dir: str, model_id: str):
    return AutoTokenizer.from_pretrained(
        f"{model_dir}/{model_id}_tokenizer", local_files_only=True
    )


def train(
    model: torch.nn.Module,
    config: wandb.Config,
    comments_text,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    loss_fn = torch.nn.CrossEntropyLoss()
    assert wandb.run is not None
    wandb.watch(model, loss_fn, log="all", log_freq=100)

    checkpoint_path = Path(config.model_dir) / wandb.run.name
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
        correct_ct = 0

        print(f"Epoch {epoch}:")
        for step, (_, comments, labels) in enumerate(train_loader):
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(comments)
            logits = outputs.logits
            train_loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(comments)
            correct_ct += (torch.argmax(logits, dim=1) == labels).sum().item()
            if step % 64 == 0:
                wandb.log({"train/loss": train_loss}, step=example_ct)

        train_accuracy = correct_ct / len(train_loader.dataset)  # type: ignore
        val_loss, logits, labels = _evaluate(
            model,
            comments_text,
            loss_fn,
            val_loader,
            log_examples=(epoch % config.checkpoint_every_n == 0),
            log_n_worst=config.log_n_worst,
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
            epoch > 0
            and config.checkpoint_every_n is not None
            and epoch % config.checkpoint_every_n == 0
        ):
            torch.save(
                model_latest,
                checkpoint_path / f"{config.model_name}_epoch={epoch}.ckpt",
            )

        (
            f1,
            auprc,
            val_accuracy,
            precision_0,
            precision_1,
            recall_0,
            recall_1,
        ) = compute_metrics(logits, labels)

        wandb.log(
            {
                "validation/loss": val_loss,
                "validation/accuracy": val_accuracy,
                "validation/f1": f1,
                "validation/auprc": auprc,
                "validation/precision_non-toxic": precision_0,
                "validation/precision_toxic": precision_1,
                "validation/recall_non-toxic": recall_0,
                "validation/recall_toxic": recall_1,
                "validation/pr_curve": wandb.plot.pr_curve(
                    labels, torch.softmax(logits, dim=1)
                ),
                "train/accuracy": train_accuracy,
                "epoch": epoch,
            },
            step=example_ct,
        )

        print(
            f"Train Acc: {train_accuracy:.2f}, Val Loss: {val_loss:2f}, Val Acc: {val_accuracy:.2f}"
        )
        print()

    torch.save(model_latest, checkpoint_path / f"{config.model_name}_latest.ckpt")
    torch.save(model_best, checkpoint_path / f"{config.model_name}_best.ckpt")
    if config.log_model_to_wandb:
        model_artifact = wandb.Artifact(
            "toxicity-baseline",
            type="model",
            metadata=dict(config),
        )
        model_artifact.add_file(checkpoint_path / f"{config.model_name}_best.ckpt")
        wandb.log_artifact(model_artifact, aliases=["best", "latest"])


def _evaluate(
    model: torch.nn.Module,
    comments_text,
    loss_fn: Callable,
    loader: DataLoader,
    log_examples: bool,
    log_n_worst: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.inference_mode():
        loss = 0
        comments_batched, logits_batched, labels_batched = [], [], []

        for idx, comments, labels in loader:
            comments, labels = comments.to(device), labels.to(device)

            outputs = model(comments)
            logits = outputs.logits

            loss += loss_fn(logits, labels).item() * labels.size(0)

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
            log_sample_predictions(
                [comments[i] for i in top_n_indices],
                predictions,
                labels[top_n_indices],
                logits.softmax(dim=1),
            )

        total = len(loader.dataset)  # type: ignore
        return loss / total, logits, labels


def log_sample_predictions(comments, predictions, true_labels, probabilities):
    table = wandb.Table(
        columns=["comment", "prediction", "target", "prob_non_toxic", "prob_toxic"]
    )
    for text, pred, targ, prob in zip(
        comments,
        predictions.to("cpu"),
        true_labels.to("cpu"),
        probabilities.to("cpu"),
    ):
        table.add_data(text, pred, targ, *prob.numpy())
    wandb.log({"predictions_table": table}, commit=False)


def compute_metrics(logits, labels):
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")(
        logits, labels
    )
    auprc = torchmetrics.AveragePrecision(task="multiclass", num_classes=2)(
        logits, labels
    ).item()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)(
        logits, labels
    ).item()
    precision_0, precision_1 = torchmetrics.Precision(
        task="multiclass", num_classes=2, average="none"
    )(logits, labels)
    recall_0, recall_1 = torchmetrics.Recall(
        task="multiclass", num_classes=2, average="none"
    )(logits, labels)

    return f1, auprc, accuracy, precision_0, precision_1, recall_0, recall_1


def test(
    model: torch.nn.Module, comments_text, config: wandb.Config, loader: DataLoader
):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, logits, labels = _evaluate(
        model,
        comments_text,
        loss_fn,
        loader,
        log_examples=True,
        log_n_worst=config.log_n_worst,
    )

    f1, auprc, accuracy, precision_0, precision_1, recall_0, recall_1 = compute_metrics(
        logits, labels
    )

    print(logits.numpy())
    np.savetxt(
        config.logits_path + '/' + config.base_model_id, logits.numpy(),  
        delimiter=',', newline='\n', header='', footer='', comments=''
    )

    wandb.summary.update(
        {
            "test/loss": loss,
            "test/accuracy": accuracy,
            "test/f1": f1,
            "test/auprc": auprc,
            "test/precision_non-toxic": precision_0,
            "test/precision_toxic": precision_1,
            "test/recall_non-toxic": recall_0,
            "test/recall_toxic": recall_1,
        }
    )
