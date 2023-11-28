import os

from dataset import setup_datasets_2
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model 
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import transformers
from datasets import load_dataset
import jsonlines
import wandb
import yaml

from huggingface_hub import login
login(token='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', add_to_git_credential=True)


user = "oovcharenko" if True else "ewybitul"

config_local = {}
with open("configs/defaults.yaml") as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)
    config_local.update(base_config)
with open("configs/toxicity/defaults.yaml") as f:
    toxicity_config = yaml.load(f, Loader=yaml.FullLoader)
    config_local.update(toxicity_config)

config_local.update(
    {
        "model_directory": f"/cluster/scratch/{user}/dsl_hate_speech/models",
        "train_data": "data/processed_comments_train_v3.csv",
        "evaluation_data": "data/processed_comments_evaluation_v3.csv",
        "model": "toxicity-detection-llm",
        "early_stopping_enabled": False,
        "early_stopping_epoch": 3,
        "early_stopping_metric": "validation/loss",
        "early_stopping_threshold": 0.37,
        "epochs": 5,
    }
)
wandb.init(project="toxicity-detection-llm", config=config_local)

match wandb.config["base_model"]:
    case "Hate-speech-CNERG/dehatebert-mono-german":
        wandb.config.update(
            {"transform_remove_umlauts": True, "transform_lowercase": True},
            allow_val_change=True,
        )

config = wandb.config


model_path = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained( 
    model_path,
    load_in_8bit=True, 
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


df_train, df_eval = setup_datasets_2(config)
with jsonlines.open("data/llm/train.jsonl", mode="w") as writer:
    for row in df_train.iter_rows(named=True):
        text, label = row["comment_preprocessed_legacy"], row["toxic"]
        completion = "toxic" if label == 1 else "not toxic"
        prompt = '''Toxic comment is any kind of offensive or denigrating speech against humans based on
        their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
        This comment is {}: "{}"'''.format(
            completion, text
        )

        writer.write({"prompt": prompt})

with jsonlines.open("data/llm/eval.jsonl", mode="w") as writer:
    for row in df_eval.iter_rows(named=True):
        completion = "toxic" if label == 1 else "not toxic"
        prompt = '''Toxic comment is any kind of offensive or denigrating speech against humans based on
        their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
        This comment is {}: "{}"'''.format(
            completion, text
        )

        writer.write({"prompt": prompt})

data = load_dataset("data/llm/train")
print(data)
data = data.map(lambda samples: tokenizer(samples['prompt']), batched=True)
print(data)


trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# # Inference
# data = load_dataset("data/llm/eval")
# print(data)
# data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)
# print(data)

# batch = tokenizer("Two things are infinite: ", return_tensors='pt')

# with torch.cuda.amp.autocast():
#   output_tokens = model.generate(**batch, max_new_tokens=50)

# print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
