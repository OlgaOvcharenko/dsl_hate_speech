from pathlib import Path

# import peft
import torch
from adapters import AutoAdapterModel, MAMConfig, UniPELTConfig
from transformers import AutoModelForSequenceClassification, logging

import wandb
import wandb.plot

logging.set_verbosity_error()


def get_base_model_path(config: wandb.Config):
    print(Path(config.model_directory) / f"{config.base_model}_model")
    return Path(config.model_directory) / f"{config.base_model}_model"


class MultiClassAdapterModule(torch.nn.Module):
    def __init__(self, config: wandb.Config, local_files=True):
        super().__init__()
        if config.base_model == "dbmdz/german-gpt2":
            model = AutoAdapterModel.from_pretrained(
                get_base_model_path(config),
                problem_type="single_label_classification",
                num_labels=len(config.class_names),
                local_files_only=local_files,
            )
        else:
            model = AutoAdapterModel.from_pretrained(
                get_base_model_path(config),
                problem_type="single_label_classification",
                num_labels=2,
                local_files_only=local_files,
                hidden_dropout_prob=config.hidden_dropout_prob,
            )
        model.add_classification_head("toxicity", num_labels=2)
        match config.adapter_type:
            case "default":
                model.add_adapter("toxicity")
            case "mam":
                model.add_adapter("toxicity", config=MAMConfig())
            case "unipelt":
                model.add_adapter("toxicity", config=UniPELTConfig())
        model.train_adapter("toxicity")
        model.set_active_adapters("toxicity")
        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
