from transformers import AutoModelForSequenceClassification, AutoTokenizer


def save_model_local(
    model_id,
    num_labels=2,
    problem_type="single_label_classification",
    model_dir="models",
):
    model_path = f"{model_dir}/{model_id}_model"
    tokenizer_path = f"{model_dir}/{model_id}_tokenizer"
    AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    ).save_pretrained(model_path)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(tokenizer_path)


save_model_local(model_id="TheBloke/Mistral-7B-v0.1-AWQ")
