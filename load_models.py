import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys


def save_model_local(model_id: str, model_path: str, tokenizer_path, num_labels=2):
    AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    ).save_pretrained(model_path)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(tokenizer_path)


def load_models(models: list):
    for model in models:
        save_model_local(model, "models/"+model+'_model', "models/"+model+'_tokenizer')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    args = parser.parse_args()
    load_models(args.models)
    