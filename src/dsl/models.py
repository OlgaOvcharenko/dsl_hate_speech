from pathlib import Path

import torch
import wandb.plot
from transformers import AutoModelForSequenceClassification, logging
from transformers.adapters import BertAdapterModel

logging.set_verbosity_error()

import wandb


def get_base_model_path(config: wandb.Config):
    return Path(config.model_directory) / f"{config.base_model}_model"


def _freeze_layers(model, layers):
    for name, param in model.named_parameters():
        for i in layers:
            if f"encoder.layer.{i}." in name:
                param.requires_grad = False


class MultiLabelModule(torch.nn.Module):
    def __init__(self, config: wandb.Config, local_files=True):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            get_base_model_path(config),
            problem_type="multi_label_classification",
            num_labels=len(config.class_names),
            local_files_only=local_files,
        )
        _freeze_layers(self.model, config.layers_to_freeze)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )


class MultiClassAdapterModule(torch.nn.Module):
    def __init__(self, config: wandb.Config, local_files=True):
        super().__init__()
        model = BertAdapterModel.from_pretrained(
            get_base_model_path(config),
            problem_type="single_label_classification",
            num_labels=2,
            local_files_only=local_files,
        )
        # FIXME We are hardcoding the adapter name here
        model.add_classification_head("toxicity", num_labels=2)
        model.add_adapter("toxicity")
        model.train_adapter("toxicity")
        model.set_active_adapters("toxicity")
        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )


class MultiClassModule(torch.nn.Module):
    def __init__(self, config: wandb.Config, local_files=True):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            get_base_model_path(config),
            problem_type="single_label_classification",
            num_labels=len(config.class_names),
            local_files_only=local_files,
        )
        _freeze_layers(self.model, config.layers_to_freeze)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
