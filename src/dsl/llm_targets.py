import os

import numpy as np

from dataset import setup_datasets_2, setup_datasets_targets_only
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model 
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
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

config_local = wandb.config


model_path = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained( 
    model_path,
    # load_in_8bit=True, 
    device_map='auto',
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=False, 
        load_in_4bit=True
    ),
    torch_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

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


df_train, df_eval = setup_datasets_targets_only(config_local)
with jsonlines.open("data/llm_target/train.jsonl", mode="w") as writer:
    for row in df_train.iter_rows(named=True):
        text = row["comment_preprocessed_legacy"]
        target_categories = ["gender", "age", "sexuality", "religion", "nationality", "disability", "social_status", "political_views", "appearance", "other"]
        
        curr_targets = ""
        for val in target_categories:
           if row[val] == 1:
              val_fix = val.replace("_", " ")
              curr_targets = curr_targets + val_fix + ", "
        
        if len(curr_targets) > 2:
           curr_targets = curr_targets[:-2]


        prompt = '''INSTRUCTION: Hate speech is any kind of offensive or denigrating speech against humans based on their identity. 
        Hate speech can be targeted towards gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, or other characteristic.
        \nINPUT: What is 1 or more targets of this comment "{}"? 
        Use only the following targets: gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, other. \nOUTPUT: "{}".'''.format(
            text, curr_targets
        )

        writer.write({"text": prompt})

with jsonlines.open("data/llm_target/validation.jsonl", mode="w") as writer:
    for row in df_eval.iter_rows(named=True):
        text = row["comment_preprocessed_legacy"]
        target_categories = ["gender", "age", "sexuality", "religion", "nationality", "disability", "social_status", "political_views", "appearance", "other"]
        
        curr_targets = ""
        for val in target_categories:
           if row[val] == 1:
              val_fix = val.replace("_", " ")
              curr_targets = curr_targets + val_fix + ", "
        
        if len(curr_targets) > 2:
           curr_targets = curr_targets[:-2]


        prompt = '''INSTRUCTION: Hate speech is any kind of offensive or denigrating speech against humans based on their identity. 
        Hate speech can be targeted towards gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, or other characteristic.
        \nINPUT: What is 1 or more targets of this comment "{}"? 
        Use only the following targets: gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, other. \nOUTPUT: "{}".'''.format(
            text, curr_targets
        )

        writer.write({"text": prompt})

with jsonlines.open("data/llm_target/test.jsonl", mode="w") as writer:
    for row in df_eval.iter_rows(named=True):
        text = row["comment_preprocessed_legacy"]

        prompt = '''INSTRUCTION: Hate speech is any kind of offensive or denigrating speech against humans based on their identity. 
        Hate speech can be targeted towards gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, or other characteristic.
        \nINPUT: What is 1 or more targets of this comment "{}"? 
        Use only the following targets: gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, other.'''.format(
            text
        )

        writer.write({"text": prompt})

data = load_dataset("data/llm_target/")
print(data)
data = data.map(lambda samples: tokenizer(samples['text']), batched=True)
print(data)


trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    eval_dataset=data['validation'],
    args=transformers.TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        # max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
with torch.autocast("cuda"):
    trainer.train()
    trainer.evaluate()

p, l, m = trainer.predict(data["test"])
np.savetxt("data/predict_targets.csv", p, delimiter = ",")

# text = "Gute Idee und die Hassprediger auch gleich ausweisen und keine dieser Sorte mehr ins Land lassen. Gleichzeitig aber auch ein europaweites Verzeichnis pädophiler Pfaffen und diesen ein Berufsverbot auferlegen. Wird endlich Zeit dass in diesen Religionen aufgeräumt wird!"
# prompt = '''INSTRUCTION: Toxic comment is any kind of offensive or denigrating speech against humans based on
#         their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
#         \nINPUT: Is this comment toxic "{}"? Answer only yes or no.'''.format(
#             text
#         )
# batch = tokenizer(prompt, return_tensors='pt')

# with torch.cuda.amp.autocast():
#   output_tokens = model.generate(**batch, max_new_tokens=50)

# print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
