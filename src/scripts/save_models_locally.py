from transformers import AutoModelForSequenceClassification, AutoTokenizer


def save_model_local(
    model_id,
    model_path,
    tokenizer_path,
    num_labels=2,
    problem_type="single_label_classification",
):
    AutoModelForSequenceClassification.from_pretrained(
        model_id,
        #num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    ).save_pretrained(model_path)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(tokenizer_path)


save_model_local(
    model_id="bert-base-german-cased",
    model_path="models/bert-base-uncased_model",
    tokenizer_path="models/bert-base-uncased_tokenizer",
    #num_labels=2,
)
