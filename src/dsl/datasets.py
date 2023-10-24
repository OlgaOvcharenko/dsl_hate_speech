from typing import Callable, Optional

import numpy as np
import polars as pr
import polars.selectors as cs
import torch
import wandb.plot
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import wandb


def _get_tokenizer(model_dir: str, model_id: str, local_files=True):
    return AutoTokenizer.from_pretrained(
        f"{model_dir}/{model_id}_tokenizer", local_files_only=local_files
    )


class CommentDataset(Dataset):
    def __init__(self, comments: np.ndarray, labels: torch.Tensor, tokenizer: Callable):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.comments)

    def _tokenize(self, comments):
        return self.tokenizer(
            comments.tolist(), padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]

    # def __getitem__(self, index):
    #     return index, self._tokenize(self.comments[index]), self.labels[index]

    def __getitems__(self, indices):
        tokenized_comments = self._tokenize(self.comments[indices])
        return list(
            (i, c, l.item() if len(l.shape) == 1 else l.float())
            for i, c, l in zip(indices, tokenized_comments, self.labels[indices])
        )


def load_data(config: wandb.Config, debug_subset: Optional[int] = None):
    df = pr.read_csv(config.data_path)
    if debug_subset is not None:
        df = df.head(debug_subset)
    if len(config.class_names) == 2:
        cols = df.select(cs.by_name(config.class_names[1]))
    else:
        cols = df.select(cs.by_name(config.class_names))
    labels = torch.tensor(cols.to_numpy())[:debug_subset]
    return df, df["comment"], labels


def setup_datasets(
    comments: pr.Series,
    labels: torch.Tensor,
    config: wandb.Config,
    test_split: float = 0.2,
    val_split: float = 0.2,
):
    train_data, test_data, train_labels, test_labels = train_test_split(
        comments.to_numpy(),
        labels.numpy(),
        test_size=test_split,
        shuffle=True,
        stratify=labels.numpy() if len(config.class_names) == 2 else None,
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data,
        train_labels,
        test_size=val_split / (1 - test_split),
        shuffle=True,
        stratify=train_labels if len(config.class_names) == 2 else None,
    )

    tokenizer = _get_tokenizer(config.model_dir, config.base_model_id)
    train_dataset = CommentDataset(train_data, torch.tensor(train_labels), tokenizer)
    val_dataset = CommentDataset(val_data, torch.tensor(val_labels), tokenizer)
    test_dataset = CommentDataset(test_data, torch.tensor(test_labels), tokenizer)

    return train_dataset, val_dataset, test_dataset


def setup_loader(data: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
