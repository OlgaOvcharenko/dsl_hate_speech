# %%

import re
import time
from typing import Callable

import emoji
import numpy as np
import polars as pl
import preprocessor as p
import torch
import tqdm
from dsl.datasets import _get_tokenizer, setup_loader
from dsl.models import MultiClassAdapterModule
from torch.utils.data import Dataset

import wandb

# %%
config = {
    "model_directory": "./models",
    # TODO These depend on the concrete model we use
    "base_model": "xlm-roberta-large",
    "adapter_type": "default",
    "hidden_dropout_prob": 0.1,
    "transform_clean": True,
    "transform_lowercase": False,
    "transform_remove_umlauts": False,
    # TODO These depend on the concrete dataset we use
    "trained_model_path": "models/checkpoints/targeted-detection-baseline_best.ckpt",
    "task": "hate_speech_split",
    "inference_data": "out/oct23_with_toxicity.csv",
    "decision_threshold": 0.5,
}

wandb.init(project="inference", config=config)
config = wandb.config


# %% ------------------------------ Model loading ----------------------------- #

model = MultiClassAdapterModule(config)
tokenizer = _get_tokenizer(config.model_directory, config.base_model)

checkpoint = torch.load(
    config.trained_model_path, map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint["model"])

# %% ---------------------------- Data loading ------------------------------- #

df = pl.read_csv(config.inference_data)
comments = df["body"].to_numpy()
ids = df["id"].to_numpy()

if config.task == "hate_speech_split":
    is_toxic = df["toxic_probability"] > config.decision_threshold
    comments = comments[is_toxic]
    ids = ids[is_toxic]


class InferenceDataset(Dataset):
    def __init__(
        self,
        comments: np.ndarray,
        ids: np.ndarray,
        tokenizer: Callable,
        lowercase: bool,
        tweet_clean: bool,
        remove_umlauts: bool,
    ):
        self.ids = ids
        self.comments = self._preprocess(
            comments,  # type: ignore
            lowercase=lowercase,
            tweet_clean=tweet_clean,
            remove_umlauts=remove_umlauts,
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.comments)

    def _preprocess(
        self,
        comments: list,
        lowercase: bool,
        tweet_clean: bool,
        remove_umlauts: bool,
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
            comments.tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

    def __getitems__(self, indices):
        tokenized_comments = self._tokenize(self.comments[indices])
        return [
            (ids[indices[i]], tokenized_comments[i])
            for i in range(len(indices))
        ]


dataset = InferenceDataset(
    comments=comments,
    ids=ids,
    tokenizer=tokenizer,
    lowercase=config.transform_lowercase,
    tweet_clean=config.transform_clean,
    remove_umlauts=config.transform_remove_umlauts,
)

# %%
loader = setup_loader(dataset, shuffle=False, batch_size=16, num_workers=0)

# %% ---------------------------- Inference ---------------------------------- #

device = torch.device("mps")
model.to(device)
model.eval()

example_ct = 0
start_time = time.time()
with torch.inference_mode():
    ids_batched = []
    labels_batched = []

    for idx, comments in tqdm.tqdm(loader):
        example_ct += len(comments)
        comments = comments.to(device)

        outputs = model(input_ids=comments)
        preds = outputs.logits.softmax(dim=1)
        labels_batched.append(preds)
        ids_batched.append(idx)

        wandb.log(
            {"throughput": example_ct / (time.time() - start_time)},
            step=example_ct,
        )

    labels = torch.cat(labels_batched, dim=0)
    ids = [id for ids in ids_batched for id in ids]

# %%
if config.task == "hate_speech_split":
    pred_df = pl.DataFrame(
        {"id": ids, "hate_speech_probability": pl.Series(labels[:, 1].tolist())}
    )
    df = df.join(pred_df, on="id", how="left", validate="1:1")
    df = df.with_columns(
        hate_speech_probability=pl.col("hate_speech_probability").fill_null(0)
    )
    df.write_csv("out/oct23_with_hate_speech.csv")
elif config.task == "toxicity":
    df = df.with_columns(toxic_probability=pl.Series(labels[:, 1].tolist()))
    df.write_csv("out/oct23_with_toxicity.csv")

# %%
