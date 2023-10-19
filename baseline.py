import os
import random
from typing import Callable, Optional

import numpy as np
import polars as pr
import torch
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# %%
SEED = 42
os.environ["PL_GLOBAL_SEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL = "models_local/" + "Hate-speech-CNERG/dehatebert-mono-german" + "_model"
TOKENIZER = "models_local/" + "Hate-speech-CNERG/dehatebert-mono-german" + "_tokenizer"


def save_model_local():
    AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, local_files_only=True
    ).save_pretrained(f"models_local/{MODEL}_model")
    AutoTokenizer.from_pretrained(
        TOKENIZER
    ).save_pretrained(f"models_local/{MODEL}_tokenizer")


# %%
class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, data: pr.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data["kommentar"])

    def __getitem__(self, index):
        return (
            self.data["kommentar"][index],
            torch.tensor(self.data["label"][index], dtype=torch.float32)
        )


def setup_data(debug_subset: Optional[int] = None):
    data_path = "./data/all_DEFR_comments_27062022.csv"

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


def setup_dataloader(data: torch.utils.data.Dataset, shuffle: bool, batch_size: int = 16):
    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle
    )


# %%
class BERTModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=2, local_files_only=True
        )

        freeze_sublayers = [
            "encoder.layer.0.",
            "encoder.layer.1.",
            "encoder.layer.2.",
            "encoder.layer.3.",
            "encoder.layer.4.",
            "encoder.layer.5.",
            "encoder.layer.6.",
            "encoder.layer.7.",
            "encoder.layer.8.",
            "encoder.layer.9.",
            "encoder.layer.10.",
        ]
        for name, param in self.model.named_parameters():
            for freeze_layer in freeze_sublayers:
                if freeze_layer in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)


def train_loop(model: torch.nn.Module, epochs: int, train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader, verbose: bool = False):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER
    )

    model.to(device)
    # tokenizer.to(device)

    for i in range(epochs):
        model.train()
        losses = 0
        print(f"Epoch {i}:")
        for batch_X, batch_Y in tqdm.tqdm(train_loader):
            batch_X = tokenizer.batch_encode_plus(
                list(batch_X), padding=True, truncation=True, return_tensors="pt"
            )["input_ids"].to(device)

            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()

            outputs = model(batch_X)

            logits = outputs.logits
            batch_Y = torch.squeeze(batch_Y).long()
            loss = loss_fn(logits, batch_Y)

            loss.backward()
            optimizer.step()

            losses += loss.item()

        if verbose:
            print(f"Train loss after {epochs: 04d} epochs: {losses / ((i + 1) * len(train_loader)): 0.4f}")

        if i % 10 == 0:
            validation(model, tokenizer, loss_fn, val_loader, epochs, device)

    return model, tokenizer


def validation(model: torch.nn.Module, tokenizer, loss_fn: Callable, val_dataloader: torch.utils.data.DataLoader,
               epochs: int, device: str = "cpu"):
    with torch.inference_mode():
        model.eval()

        val_loss = 0
        targets, outs = [], []
        print("Validation:")
        for valid_X, valid_Y in tqdm.tqdm(val_dataloader):
            valid_X = tokenizer.batch_encode_plus(
                list(valid_X), padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]
            valid_X, valid_Y = valid_X.to(device), valid_Y.to(device)

            outputs = model(valid_X)
            logits = outputs.logits
            # logits = torch.argmax(logits, dim=1)

            # outs.extend(logits > 0)
            outs.extend(np.argmax(logits.cpu().numpy(), axis=1))

            targets.extend(valid_Y)
            valid_Y = valid_Y.unsqueeze(0)
            valid_Y = valid_Y[0, :]
            valid_Y = torch.squeeze(valid_Y).long()
            val_loss += loss_fn(logits, valid_Y).item()
        val_loss = val_loss / len(val_dataloader)
        # scheduler.step(val_loss)
        print(f"  Val loss after {epochs: 04d} epochs: {val_loss: 0.4f}")


def test(model: torch.nn.Module, tokenizer, test_dataloader: torch.utils.data.DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.inference_mode():
        model.eval()
        pred_Y, test_Y_all = [], []

        print("Test:")
        for test_X, test_Y in tqdm.tqdm(test_dataloader):
            test_X = tokenizer.batch_encode_plus(
                list(test_X), padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]
            test_X, test_Y = test_X.to(device), test_Y.to(device)

            outputs = model(test_X)
            logits = outputs.logits
            pred_Y.extend(np.argmax(logits.cpu().numpy(), axis=1))
            test_Y_all.extend(test_Y.cpu().numpy())

        pred_Y = np.array(pred_Y)
        test_Y_all = np.array(test_Y_all)

        acc = sum(pred_Y == test_Y_all) / len(test_Y_all)
        print(f"Accuracy is {acc}")

        return pred_Y, test_Y_all
