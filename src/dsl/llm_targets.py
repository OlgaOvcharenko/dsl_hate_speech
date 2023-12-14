import os

import numpy as np

from dataset import setup_datasets_2, setup_datasets_targets_only
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
import pandas as pd

from huggingface_hub import login
login(token='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', add_to_git_credential=True)


user = "oovcharenko" if True else "ewybitul"

config_local = {}
with open("configs/defaults.yaml") as f:
    base_config = yaml.load(f, Loader=yaml.FullLoader)
    config_local.update(base_config)
with open("configs/targets/defaults.yaml") as f:
    toxicity_config = yaml.load(f, Loader=yaml.FullLoader)
    config_local.update(toxicity_config)

config_local.update(
    {
        "model_directory": f"/cluster/scratch/{user}/dsl_hate_speech/models",
        "train_data": "data/processed_training_main_v4.csv",
        "evaluation_data": "data/processed_evaluation_main_v4.csv",
        "model": "toxicity-detection-llm",
        "early_stopping_enabled": False,
        "early_stopping_epoch": 3,
        "early_stopping_metric": "validation/loss",
        "early_stopping_threshold": 0.37,
        "epochs": 5,
    }
)
wandb.init(project="targets-detection-llm", config=config_local)

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
    #device_map='auto',
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    device_map={'':torch.cuda.current_device()},
    torch_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # reduce number of stored activations
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

df_train = setup_datasets_targets_only(config_local, file=config_local.train_data)
with jsonlines.open("data/llm_target/train.jsonl", mode="w") as writer:
    for row in df_train.iter_rows(named=True):
        text = row["comment_preprocessed_legacy"]
        target_categories = ["gender", "age", "sexuality", "religion", "nationality", "disability", "social_status", "political_views", "appearance", "other"]
        if len(text) < 500:
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
            Use only the following targets: gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, other. \nOUTPUT: {}.'''.format(
                text, curr_targets
            )

            writer.write({"text": prompt})

# df_eval = setup_datasets_targets_only(config_local, file=config_local.evaluation_data)
# with jsonlines.open("data/llm_target/validation.jsonl", mode="w") as writer:
#     for row in df_eval.iter_rows(named=True):
#         text = row["comment_preprocessed_legacy"]
#         target_categories = ["gender", "age", "sexuality", "religion", "nationality", "disability", "social_status", "political_views", "appearance", "other"]
#         if len(text) < 500:
#             curr_targets = ""
#             for val in target_categories:
#                 if row[val] == 1:
#                     val_fix = val.replace("_", " ")
#                     curr_targets = curr_targets + val_fix + ", "
            
#             if len(curr_targets) > 2:
#                 curr_targets = curr_targets[:-2]


#             prompt = '''INSTRUCTION: Hate speech is any kind of offensive or denigrating speech against humans based on their identity. 
#             Hate speech can be targeted towards gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, or other characteristic.
#             \nINPUT: What is 1 or more targets of this comment "{}"? 
#             Use only the following targets: gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, other. \nOUTPUT: {}.'''.format(
#                 text, curr_targets
#             )

#             writer.write({"text": prompt})

# with jsonlines.open("data/llm_target/test.jsonl", mode="w") as writer:
#     for row in df_eval.iter_rows(named=True):
#         text = row["comment_preprocessed_legacy"]

#         prompt = '''INSTRUCTION: Hate speech is any kind of offensive or denigrating speech against humans based on their identity. 
#         Hate speech can be targeted towards gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, or other characteristic.
#         \nINPUT: What is 1 or more targets of this comment "{}"? 
#         Use only the following targets: gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, other.'''.format(
#             text
#         )

#         writer.write({"text": prompt})

data = load_dataset("data/llm_target/")
data = data.map(lambda samples: tokenizer(samples['text']), batched=True)

training_args = transformers.TrainingArguments(
        num_train_epochs=0.01,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        # max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs_targets_new'
    )

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    # eval_dataset=data['validation'],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("parallel_mode: ", training_args.parallel_mode)
print("n_gpus: ", training_args.n_gpu)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
with torch.autocast("cuda"):
    trainer.train(resume_from_checkpoint=False)
    # res = trainer.evaluate()
    # print(res)

    model.save_pretrained("outputs_targets_new/")

    # p, l, m = trainer.predict(data["test"])
    # np.savetxt("data/predict_binary.csv", p, delimiter = ",")

# Inference
model.config.use_cache = True
device = torch.device('cuda')

df_eval = setup_datasets_targets_only(config_local, file=config_local.evaluation_data)

results, targets_cat = [], []

for row in df_eval.iter_rows(named=True):
    text = row["comment_preprocessed_legacy"]
    target_categories = ["gender", "age", "sexuality", "religion", "nationality", "disability", "social_status", "political_views", "appearance", "other"]
    curr_targets = ""
    for val in target_categories:
        if row[val] == 1:
            targets_cat.append(val)
            val_fix = val.replace("_", " ")
            curr_targets = curr_targets + val_fix + ", "
    
    if len(curr_targets) > 2:
        curr_targets = curr_targets[:-2]


    prompt = '''INSTRUCTION: Hate speech is any kind of offensive or denigrating speech against humans based on their identity. 
    Hate speech can be targeted towards gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, or other characteristic.
    \nINPUT: What is 1 or more targets of this comment "{}"? 
    Use only the following targets: gender, age, sexuality, religion, nationality, disability, social status, political views, appearance, other.'''.format(
        text
    )


    batch = tokenizer(prompt, return_tensors='pt')

    with torch.cuda.amp.autocast():
        batch = batch.to(device)
        output_tokens = model.generate(**batch, max_new_tokens=50)
        res = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print('\n\n', res)

        results.append(res)

df_res = pd.DaraFrame(results)
df_res["cat"] = targets_cat

# df_res.to_csv("outputs_targets/results_main_eval.csv", sep=",", index=False)
