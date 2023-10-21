import math
import os
import random
from typing import Callable, Optional

import numpy as np
import polars as pr
import torch
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb

# %%
SEED = 42
os.environ["PL_GLOBAL_SEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_DIR = "models_local"


def save_model_local(num_labels=2):
    AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=num_labels
    ).save_pretrained(MODEL)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(TOKENIZER)


# %%
class CommentDataset(torch.utils.data.Dataset):
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
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        CommentDataset(data), [train_size, val_size, test_size]
    )
    return train_dataset, val_dataset, test_dataset


def setup_loader(data: torch.utils.data.Dataset, shuffle: bool, batch_size: int = 16):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)


class PretrainedModule(torch.nn.Module):
    def __init__(self, config: wandb.Config):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            f"{MODEL_DIR}/{config.model_id}_model", num_labels=2, local_files_only=True
        )

        for name, param in self.model.named_parameters():
            for i in range(11):
                if f"encoder.layer.{i}." in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)


def get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(
        f"{MODEL_DIR}/{model_id}_tokenizer", local_files_only=True
    )


def train(
    model: torch.nn.Module,
    config: wandb.Config,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
):
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.watch(model, loss_fn, log="all", log_freq=100)
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps_per_epoch = math.ceil(len(train_loader.dataset) / config.batch_size)

    tokenizer = get_tokenizer(config.model_id)
    model.to(device)
    # tokenizer.to(device)

    example_ct = 0
    for epoch in range(config.epochs):
        model.train()
        print(f"Epoch {epoch}:")
        for step, (comments, labels) in enumerate(tqdm.tqdm(train_loader)):
            comments = tokenizer.batch_encode_plus(
                list(comments), padding=True, truncation=True, return_tensors="pt"
            )["input_ids"].to(device)
            comments, labels = comments.to(device), labels.to(device).long()
            if torch.cuda.is_available():
                comments = comments.cuda()
                labels = labels.cuda()

            outputs = model(comments)
            logits = outputs.logits
            train_loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            example_ct += len(comments)
            train_metrics = {
                "train/train_loss": train_loss,
                "train/example_ct": example_ct,
            }
            last_step = step == steps_per_epoch - 1
            if not last_step:
                wandb.log(train_metrics)

        val_loss, val_accuracy = validate(
            model,
            tokenizer,
            loss_fn,
            val_loader,
            device,
            log_examples=(epoch == config.epochs - 1),
        )
        val_metrics = {"val/val_loss": val_loss, "val/val_accuracy": val_accuracy}

        wandb.log({**train_metrics, **val_metrics})
        print(
            f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {val_accuracy:.2f}"
        )
        print()


def validate(
    model: torch.nn.Module,
    tokenizer,
    loss_fn: Callable,
    val_dl: torch.utils.data.DataLoader,
    device: str = "cpu",
    log_examples=False,
):
    model.eval()
    with torch.inference_mode():
        loss = 0
        correct_ct = 0
        print("Validation:")
        for batch_idx, (comments_text, labels) in enumerate(tqdm.tqdm(val_dl)):
            comments = tokenizer.batch_encode_plus(
                list(comments_text), padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]
            comments, labels = comments.to(device), labels.to(device).long()

            outputs = model(comments)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            # logits = torch.argmax(logits, dim=1)

            # outs.extend(logits > 0)
            # outs.extend(np.argmax(logits.cpu().numpy(), axis=1))

            # targets.extend(toxicity_labels)
            # toxicity_labels = toxicity_labels.unsqueeze(0)
            # toxicity_labels = toxicity_labels[0, :]
            # toxicity_labels = torch.squeeze(toxicity_labels).long()
            loss += loss_fn(logits, labels).item() * labels.size(0)
            correct_ct += (predictions == labels).sum().item()
            if batch_idx == 0 and log_examples:
                log_sample_predictions(
                    comments_text, predictions, labels, logits.softmax(dim=1)
                )

        return loss / len(val_dl.dataset), correct_ct / len(val_dl.dataset)


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


def test(
    model: torch.nn.Module, config: wandb.Config, loader: torch.utils.data.DataLoader
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(config.model_id)

    model.eval()
    with torch.inference_mode():
        pred_Y, test_Y_all = [], []

        print("Test:")
        for comments, labels in tqdm.tqdm(loader):
            comments = tokenizer.batch_encode_plus(
                list(comments), padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]
            comments, labels = comments.to(device), labels.to(device).long()

            outputs = model(comments)
            logits = outputs.logits
            pred_Y.extend(np.argmax(logits.cpu().numpy(), axis=1))
            test_Y_all.extend(labels.cpu().numpy())

        pred_Y = np.array(pred_Y)
        test_Y_all = np.array(test_Y_all)

        acc = sum(pred_Y == test_Y_all) / len(test_Y_all)
        print(f"Test Accuracy: {acc}")

        return pred_Y, test_Y_all
