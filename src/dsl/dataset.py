import re
from typing import Callable, Optional
import emoji
import numpy as np
import polars as pr
import polars.selectors as cs
import preprocessor as p
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import wandb


def _get_tokenizer(model_directory: str, model_id: str, local_files=True):
    return AutoTokenizer.from_pretrained(
        f"{model_directory}/{model_id}_tokenizer", local_files_only=local_files
    )


class CommentDataset(Dataset):
    def __init__(
        self,
        comments: np.ndarray,
        labels: torch.Tensor,
        tokenizer: Callable,
        lowercase: bool,
        tweet_clean: bool,
        remove_umlauts: bool,
    ):
        self.comments = self._preprocess(
            comments,  # type: ignore
            lowercase=lowercase,
            tweet_clean=tweet_clean,
            remove_umlauts=remove_umlauts,
        )
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.comments)

    def _preprocess(
        self, comments: list, lowercase: bool, tweet_clean: bool, remove_umlauts: bool
    ):
        def _preprocess_val(val: str):
            p.set_options(p.OPT.URL, p.OPT.MENTION)  # type: ignore
            if tweet_clean:
                val = p.clean(val)
            if lowercase:
                val = val.lower()
            if remove_umlauts:
                val = (
                    val.replace("ü", "ue")
                    .replace("ä", "ae")
                    .replace("ö", "oe")
                    .replace("Ü", "Ue")
                    .replace("Ä", "Ae")
                    .replace("Ö", "Oe")
                )
            val = emoji.demojize(val, language="de")
            val = val.replace("\n", " ")
            val = re.sub(r"\s+", " ", val)
            val = re.sub(r"<\/?[^>]*>", "", val)
            val = val.strip()
            return val

        return np.array([_preprocess_val(str(comment)) for comment in comments])

    def _tokenize(self, comments):
        return self.tokenizer(
            comments.tolist(), padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]

    def __getitems__(self, indices):
        tokenized_comments = self._tokenize(self.comments[indices])
        return list(
            (i, c, l.item() if l.shape[0] == 1 else l.float())
            for i, c, l in zip(indices, tokenized_comments, self.labels[indices])
        )


def _load_data(path: str, class_names: list[str]):
    df = pr.read_csv(path)
    if len(class_names) == 2:
        cols = df.select(cs.by_name(class_names[1]))
    else:
        cols = df.select(cs.by_name(class_names))
    labels = torch.tensor(cols.to_numpy())
    return df, df["comment"].to_numpy(), labels


def _split(comments, labels, config, size=None):
    if size is None:
        size = config.validation_split
    if len(config.class_names) == 2:
        stratifier = StratifiedShuffleSplit(
            n_splits=1, test_size=size, random_state=config.seed
        )
    else:
        stratifier = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=size, random_state=config.seed
        )

    train_idx, val_idx = next(stratifier.split(comments, labels))
    return comments[train_idx], labels[train_idx], comments[val_idx], labels[val_idx]


def _split_2(comments, labels, config, size=None):
    if size is None:
        size = config.validation_split
    if len(config.class_names) == 2:
        stratifier = StratifiedShuffleSplit(
            n_splits=1, test_size=size, random_state=config.seed
        )
    else:
        stratifier = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=size, random_state=config.seed
        )

    train_idx, val_idx = next(stratifier.split(comments, labels))
    return train_idx, val_idx


def setup_datasets_2(config: wandb.Config):
    df, comments, labels = _load_data(config.train_data, config.class_names)
    if len(config.class_names) > 2:
        # Only train on toxic comments
        # TODO: Setup a separate Dataset class for this
        indices = df["toxic"] == 1
        df = df[indices]
        comments = comments[indices]
        labels = labels[indices, :]
    if config.dataset_portion < 1:
        _, val_idx = _split_2(comments, labels, config, size=config.dataset_portion)
        comments, labels = comments[val_idx], labels[val_idx]

    train_idx, val_idx = _split_2(comments, labels, config)

    return df[train_idx], df[val_idx]


def setup_datasets(config: wandb.Config, stage: str):
    tokenizer = _get_tokenizer(config.model_directory, config.base_model)
    if stage == "fit":
        df, comments, labels = _load_data(config.train_data, config.class_names)
        if len(config.class_names) > 2:
            # Only train on toxic comments
            # TODO: Setup a separate Dataset class for this
            indices = df["toxic"] == 1
            df = df[indices]
            comments = comments[indices]
            labels = labels[indices, :]
        if config.dataset_portion < 1:
            _, _, comments, labels = _split(
                comments, labels, config, size=config.dataset_portion
            )

        train_c, train_l, val_c, val_l = _split(comments, labels, config)
        train_dataset = CommentDataset(
            train_c,
            train_l,
            tokenizer,
            lowercase=config.transform_lowercase,
            tweet_clean=config.transform_clean,
            remove_umlauts=config.transform_remove_umlauts,
        )
        val_dataset = CommentDataset(
            val_c,
            val_l,
            tokenizer,
            lowercase=config.transform_lowercase,
            tweet_clean=config.transform_clean,
            remove_umlauts=config.transform_remove_umlauts,
        )

        return df, train_dataset, val_dataset
    elif stage == "test":
        df, comments, labels = _load_data(config.evaluation_data, config.class_names)
        if len(config.class_names) > 2:
            # Only eval on toxic comments
            indices = df["toxic"] == 1
            df = df[indices]
            comments = comments[indices]
            labels = labels[indices, :]
        if config.dataset_portion < 1:
            _, _, comments, labels = _split(
                comments, labels, config, size=config.dataset_portion
            )
        return df, CommentDataset(
            comments,
            labels,
            tokenizer,
            lowercase=config.transform_lowercase,
            tweet_clean=config.transform_clean,
            remove_umlauts=config.transform_remove_umlauts,
        )


def setup_loader(data: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True
    )


def class_weights_eff_num(df: pr.DataFrame, class_names: list[str], beta: float):
    if len(class_names) == 2:
        class_counts = np.array(
            [len(df) - df[class_names[1]].sum(), df[class_names[1]].sum()]
        )
    else:
        class_counts = df.select(cs.by_name(class_names)).to_numpy().sum(axis=0)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    return torch.Tensor(weights / weights.sum() * len(class_names))


def class_weights_inverse_frequency(df: pr.DataFrame, class_names: list[str]):
    if len(class_names) == 2:
        class_counts = np.array(
            [len(df) - df[class_names[1]].sum(), df[class_names[1]].sum()]
        )
    else:
        class_counts = df.select(cs.by_name(class_names)).to_numpy().sum(axis=0)
    weights = 1 / class_counts
    return torch.Tensor(weights / weights.sum() * len(class_names))
