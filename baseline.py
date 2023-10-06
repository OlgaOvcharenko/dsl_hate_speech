# %%
import polars as pr
import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# %%
pl.seed_everything(42)


# %%
class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        return {
            "input_ids": self.data["input_ids"][index],
            "label": self.data["label"][index],
            "attention_mask": self.data["attention_mask"][index],
        }


class CommentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "./data/all_DEFR_comments_27062022.csv",
        train_batch_size=16,
        eval_batch_size=16,
        test_size=1000,
        debug_subset=None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Hate-speech-CNERG/dehatebert-mono-german"
        )
        self.data_path = data_path
        self.debug_subset = debug_subset

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.test_size = test_size

    def prepare_data(self):
        df = pr.read_csv(self.data_path, dtypes={"ArticleID": pr.Utf8, "ID": pr.Utf8})
        # TODO: Ask about the NULL values
        # TODO: Ask about the duplicates and remove them properly
        df = df.drop_nulls().unique(subset=["kommentar"])
        if self.debug_subset is not None:
            df = df.head(self.debug_subset)

        self.data = self.tokenizer.batch_encode_plus(
            df["kommentar"].to_list(),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.data["label"] = torch.tensor(df["label"])

    def setup(self, stage):
        dataset = CommentDataset(self.data)

        j = self.test_size
        data, test_data = dataset[:-j], dataset[-j:]

        self.test_dataset = CommentDataset(test_data)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            CommentDataset(data), [0.8, 0.2]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.eval_batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.eval_batch_size
        )


# %%
class BERTModule(pl.LightningModule):
    def __init__(self, lr=3e-4):
        super().__init__()
        self.optimizer_class = torch.optim.SGD
        self.lr = lr
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "statworx/bert-base-german-cased-finetuned-swiss", num_labels=1
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

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer,
        #     max_lr=self.one_cycle_max_lr,
        #     total_steps=self.one_cycle_total_steps,
        # )

        # return {
        #     "optimizer": optimizer,
        #     # "lr_scheduler": scheduler,
        #     "monitor": "validation/loss",
        # }

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    def _run_on_batch(self, batch, with_preds=False):
        x, y = batch["input_ids"], batch["label"]
        logits = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        acc = torch.sum(torch.round(torch.sigmoid(logits)) == y).float() / len(y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._run_on_batch(batch)
        self.log("train/loss", loss)
        self.log("train/acc", acc, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._run_on_batch(batch)
        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        self.log("validation/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss, acc = self._run_on_batch(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)


# %%
dm = CommentDataModule(debug_subset=1000, test_size=100)
dm.prepare_data()
dm.setup(None)

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="cpu",
    logger=WandbLogger(project="hatespeech", log_model=True),
)

# %%
model = BERTModule()
model.to("cpu")


# %%
trainer.fit(model, datamodule=dm)

# %%
