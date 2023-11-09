from dataclasses import dataclass, field
from typing import Optional

import jsonlines
import torch
import yaml
from accelerate import Accelerator
from datasets import load_dataset  # type: ignore
from peft import LoraConfig  # type: ignore
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    # GPTQConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer

import wandb
from dsl.datasets import setup_datasets_2

tqdm.pandas()


config = {}
with open("configs/defaults.yaml") as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(base_config)
with open("configs/toxicity/defaults.yaml") as f:
    toxicity_config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(toxicity_config)

config.update(
    {
        "model_directory": "/cluster/scratch/ewybitul/models",
        "train_data": "data/processed_comments_train_v3.csv",
        "evaluation_data": "data/processed_comments_evaluation_v3.csv",
        "model": "toxicity-detection-llm",
        "early_stopping_enabled": False,
        "early_stopping_epoch": 2,
        "early_stopping_metric": "validation/loss",
        "early_stopping_threshold": 0.37,
        "epochs": 4,
    }
)
wandb.init(project="toxicity-detection-llm", config=config)

match wandb.config["base_model"]:
    case "Hate-speech-CNERG/dehatebert-mono-german_labels=2":
        wandb.config.update(
            {"transform_remove_umlauts": True, "transform_lowercase": True},
            allow_val_change=True,
        )

config = wandb.config


# Step 2: Load the dataset
df_train, df_eval, _, _ = setup_datasets_2(config, stage="fit")  # type: ignore
for row in df_train.iter_rows():
    with jsonlines.open("data/llm/train.jsonl", mode="w") as writer:
        text, label = row["comment_preprocessed_legacy"], row["toxic"]
        prompt = '''Toxic comment is any kind of offensive or denigrating speech against humans based on
        their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
        Respond with yes if the following tweet is toxic, else respond with no. Do not respond with anything else."{}"'''.format(
            text
        )

        completion = " Yes" if label == 1 else " No"
        writer.write({"prompt": prompt, "completion": completion})

for row in df_eval.iter_rows():
    with jsonlines.open("data/llm/eval.jsonl", mode="w") as writer:
        text, label = row["comment_preprocessed_legacy"], row["toxic"]
        prompt = '''Toxic comment is any kind of offensive or denigrating speech against humans based on
        their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
        Respond with yes if the following tweet is toxic, else respond with no. Do not respond with anything else."{}"'''.format(
            text
        )

        completion = " Yes" if label == 1 else " No"
        writer.write({"prompt": prompt, "completion": completion})

# df_train.write_json("data/llm/train.json", row_oriented=True)
# df_eval.write_json("data/llm/eval.json", row_oriented=True)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    dataset_name: Optional[str] = field(
        default="./data/llm/", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )

    model_name: Optional[str] = field(
        default="facebook/opt-350m", metadata={"help": "the model name"}
    )
    log_with: Optional[str] = field(
        default="none", metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False, metadata={"help": "Enable `trust_remote_code`"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of logging steps"}
    )
    use_auth_token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"}
    )
    num_train_epochs: Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "the number of training steps"}
    )
    save_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Number of updates steps before two checkpoint saves"},
    )
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."}
    )
    push_to_hub: Optional[bool] = field(
        default=False, metadata={"help": "Push the model to HF Hub"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the model on HF Hub"}
    )


parser = HfArgumentParser(ScriptArguments)  # type: ignore
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = GPTQConfig(
        bits=4,
        group_size=128,
        dataset="c4",
        desc_act=False,
    )
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None


model = AutoModelForCausalLM.from_pretrained(
    "https://huggingface.co/TheBloke/Mistral-7B-v0.1-AWQ",
    # script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
)

# Step 2: Get datasets
dataset = load_dataset(script_args.dataset_name, split="train")


# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    gradient_checkpointing=script_args.gradient_checkpointing,
    # TODO: uncomment that on the next release
    # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
)


# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)
