from pathlib import Path

import torch
import wandb.plot
from transformers import AutoModelForSequenceClassification

import wandb


def get_base_model_path(config: wandb.Config):
    return Path(config.model_dir) / f"{config.base_model_id}_model"


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
