# %%
import polars as pr
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer


# %%
class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index]
        y = torch.tensor(self.labels.iloc[index], dtype=torch.float32)
        return x, y


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "./data/all_DEFR_comments_27062022.csv"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Hate-speech-CNERG/dehatebert-mono-german"
        )
        self.data_path = data_path

    def prepare_data(self):
        df = pr.read_csv(self.data_path, dtypes={"ArticleID": pr.Utf8, "ID": pr.Utf8})
        df = df.drop_nulls()  # TODO: Ask about the NULL values

        comments = self.tokenizer(
            df["kommentar"].to_list(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.tensor(df["label"])

        comments, test_comments = comments[:-1000], comments[-1000:]
        labels, test_labels = labels[:-1000], labels[-1000:]

        self.test_dataset = CommentDataset(test_comments, test_labels)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            CommentDataset(comments, labels), [0.8, 0.2]
        )


# %%
