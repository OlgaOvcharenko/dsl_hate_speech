import os

import numpy as np

from dataset import setup_datasets_2
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
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

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
    device_map={'':torch.cuda.current_device()},
    #device_map='auto',
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

# max_memory = get_balanced_memory(
#     model,
#     max_memory=None,
#     no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
#     dtype='float16',
#     low_zero=False,
# )

# device_map = infer_auto_device_map(
#     model,
#     max_memory=max_memory,
#     no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
#     dtype='float16'
# )

# model = dispatch_model(model, device_map=device_map)


df_train, df_eval = setup_datasets_2(config_local)
with jsonlines.open("data/llm/train.jsonl", mode="w") as writer:
    for row in df_train.iter_rows(named=True):
        text, label = row["comment_preprocessed_legacy"], row["toxic"]
        completion = "yes" if label == 1 else "no"
        if len(text) < 1000:
            prompt = '''INSTRUCTION: Toxic comment is any kind of offensive or denigrating speech against humans based on
            their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
            \nINPUT: Is this comment toxic "{}"? Answer only yes or no. \nOUTPUT: "{}".'''.format(
                text, completion
            )

            writer.write({"text": prompt})

with jsonlines.open("data/llm/validation.jsonl", mode="w") as writer:
    for row in df_eval.iter_rows(named=True):
        text, label = row["comment_preprocessed_legacy"], row["toxic"]
        if len(text) < 1000:
            completion = "yes" if label == 1 else "no"
            prompt = '''INSTRUCTION: Toxic comment is any kind of offensive or denigrating speech against humans based on
            their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
            \nINPUT: Is this comment toxic "{}"? Answer only yes or no. \n OUTPUT: "{}".'''.format(
                text, completion
            )

            writer.write({"text": prompt})

# with jsonlines.open("data/llm/test.jsonl", mode="w") as writer:
#     for row in df_eval.iter_rows(named=True):
#         text, label = row["comment_preprocessed_legacy"], row["toxic"]
#         completion = "yes" if label == 1 else "no"
#         prompt = '''INSTRUCTION: Toxic comment is any kind of offensive or denigrating speech against humans based on
#         their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
#         \nINPUT: Is this comment toxic "{}"? Answer only yes or no.'''.format(
#             text
#         )

#         writer.write({"text": prompt})

data = load_dataset("data/llm/")
data = data.map(lambda samples: tokenizer(samples['text']), batched=True)

training_args = transformers.TrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        # max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs_7b'
    )

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    eval_dataset=data['validation'],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("parallel_mode: ", training_args.parallel_mode)
print("n_gpus: ", training_args.n_gpu)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
with torch.autocast("cuda"):
    trainer.train(resume_from_checkpoint=True)
    res = trainer.evaluate()
    print(res)
    model.save_pretrained("outputs_7b/")

    # p, l, m = trainer.predict(data["test"])
    # np.savetxt("data/predict_binary.csv", p, delimiter = ",")

# # Inference
# data = load_dataset("data/llm/eval")
# print(data)
# data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)
# print(data)
text = "Gute Idee und die Hassprediger auch gleich ausweisen und keine dieser Sorte mehr ins Land lassen. Gleichzeitig aber auch ein europaweites Verzeichnis pädophiler Pfaffen und diesen ein Berufsverbot auferlegen. Wird endlich Zeit dass in diesen Religionen aufgeräumt wird!"
prompt = '''INSTRUCTION: Toxic comment is any kind of offensive or denigrating speech against humans based on
        their identity (e.g., based on gender, age, nationality, political views, social views, sex, disability, appearance etc.).
        \nINPUT: Is this comment toxic "{}"? Answer only yes or no.'''.format(
            text
        )
batch = tokenizer(prompt, return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
