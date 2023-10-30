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
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    ).save_pretrained(model_path)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(tokenizer_path)


# save_model_local(
#     model_id="Hate-speech-CNERG/dehatebert-mono-german",
#     model_path="models/Hate-speech-CNERG/dehatebert-mono-german_multilabel_model",
#     tokenizer_path="models/Hate-speech-CNERG/dehatebert-mono-german_multilabel_tokenizer",
#     num_labels=10,
#     problem_type="multi_label_classification",
# )
