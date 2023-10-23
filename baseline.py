from pathlib import Path
from typing import Callable, Optional

import polars as pr
import torch
import torchmetrics
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb


def save_model_local(model_id, model_path, tokenizer_path, num_labels=2):
    AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    ).save_pretrained(model_path)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(tokenizer_path)


class CommentDataset(Dataset):
    def __init__(self, data: pr.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data["kommentar"])

    def __getitem__(self, index):
        return (
            self.data["kommentar"][index],
            torch.tensor(self.data["label"][index], dtype=torch.float32),
        )


def setup_data(data_path: str, debug_subset: Optional[int] = None):
    df = pr.read_csv(data_path, dtypes={"ArticleID": pr.Utf8, "ID": pr.Utf8})

    # TODO: Ask about the NULL values
    # TODO: Ask about the duplicates and remove them properly .unique(subset=["kommentar"])
    df = df.drop_nulls()
    if debug_subset is not None:
        df = df.head(debug_subset)

    data = df.select(["kommentar", "label"])
    return data


def setup_datasets(data: pr.DataFrame, test_split: float = 0.2, val_split: float = 0.2):
    full_size = data.shape[0]
    train_size = int((1 - test_split - val_split) * full_size)
    val_size = int((full_size - train_size) * val_split)
    test_size = full_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        CommentDataset(data), [train_size, val_size, test_size]
    )
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
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    loss_fn = torch.nn.CrossEntropyLoss()
    f1_fn = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
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

    tokenizer = get_tokenizer(config.model_dir, config.base_model_id)
    model.to(device)

    example_ct = 0
    model_best, model_latest = None, None
    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        correct_ct = 0

        print(f"Epoch {epoch}:")
        for step, (comments, labels) in enumerate(train_loader):
            comments = tokenizer.batch_encode_plus(
                list(comments), padding=True, truncation=True, return_tensors="pt"
            )[
                "input_ids"
            ].to(  # type: ignore
                device
            )
            comments, labels = comments.to(device), labels.to(device).long()

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
        val_loss, val_accuracy, val_f1 = _evaluate(
            model,
            tokenizer,
            loss_fn,
            f1_fn,
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

        wandb.log(
            {
                "validation/loss": val_loss,
                "validation/accuracy": val_accuracy,
                "validation/f1": val_f1,
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
    model_artifact = wandb.Artifact(
        "toxicity-baseline",
        type="model",
        metadata=dict(config),
    )
    model_artifact.add_file(checkpoint_path / f"{config.model_name}_best.ckpt")
    wandb.log_artifact(model_artifact, aliases=["best", "latest"])


def _evaluate(
    model: torch.nn.Module,
    tokenizer,
    loss_fn: Callable,
    f1_fn: Callable,
    loader: DataLoader,
    log_examples: bool,
    log_n_worst: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.inference_mode():
        loss = 0
        correct_ct = 0
        comments_batched, logits_batched, labels_batched = [], [], []

        for comments_text, labels in loader:
            comments = tokenizer.batch_encode_plus(
                list(comments_text), padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]
            comments, labels = comments.to(device), labels.to(device).long()

            outputs = model(comments)
            logits = outputs.logits

            loss += loss_fn(logits, labels).item() * labels.size(0)
            correct_ct += (torch.argmax(logits, dim=1) == labels).sum().item()

            if log_examples:
                comments_batched.append(comments_text)
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

        f1 = f1_fn(logits, labels)

        return loss / len(loader.dataset), correct_ct / len(loader.dataset), f1  # type: ignore


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


def test(model: torch.nn.Module, config: wandb.Config, loader: DataLoader):
    loss_fn = torch.nn.CrossEntropyLoss()
    f1_fn = torchmetrics.F1Score(task="multiclass", num_classes=2, average="macro")
    tokenizer = get_tokenizer(config.model_dir, config.base_model_id)
    loss, accuracy, f1 = _evaluate(
        model,
        tokenizer,
        loss_fn,
        f1_fn,
        loader,
        log_examples=True,
        log_n_worst=config.log_n_worst,
    )
    wandb.summary.update({"test/loss": loss, "test/accuracy": accuracy, "test/f1": f1})
