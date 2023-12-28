import flair
from flair.models import TARSClassifier
from flair.data import Corpus, Sentence
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import torch


device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
flair.device = device
print(device)

def read_data(path: str):
    return pd.read_csv(path, encoding='utf-8')


def train_zero(data, comment_col):
    comments = [Sentence(s, language_code='de') for s in data[comment_col].iloc[0:1000]]  # .iloc[0:2]
    print('Created sentences.')

    tars = TARSClassifier.load('tars-base')
    classes = ["geschlecht", "alter", "sexualitat", "religion", "nationalitaet", 
            "behinderung", "sozialer status", "politische ansichten",  "aussehen"]


    res = np.zeros((len(comments), len(classes)))

    for i, x in enumerate(comments):
        tars.predict_zero_shot(x, classes, multi_label=False)
        res[i, [classes.index(label.value) for label in x.labels]] = [label.score for label in x.labels]

        tars.predict_zero_shot(x, classes, multi_label=True)
        res[i, [classes.index(label.value) for label in x.labels]] = [label.score for label in x.labels]

        if i % 100 == 0:
            print(f"{i}th iteration of {data.shape[0]}")

    print(res)
    np.savetxt("data/zero_shot.csv", res, delimiter=",")


def train_few(train, test, comment_col):
    tars = TARSClassifier.load('tars-base')
    classes = ["geschlecht", "alter", "sexualitat", "religion", "nationalitaet", 
            "behinderung", "sozialer status", "politische ansichten",  "aussehen"]
    classes_eng = [
        "gender", "age", "sexuality", "religion", "nationality", "disability", 
        "social_status", "political_views", "appearance", "other"]
    
    label_type = 'target_class'
    
    comments_train = []
    for i in range(train.shape[0]):
        labels = [classes[j] for j in range(len(classes)) if train[classes_eng[j]].iloc[i,]]

        sentence = Sentence(train[comment_col].iloc[i], language_code='de')
        [sentence.add_label(label_type, l) for l in labels]
        comments_train.append(sentence)
    
    comments_test = []
    for i in range(test.shape[0]):
        labels = [classes[j] for j in range(len(classes)) if train[classes_eng[j]].iloc[i,]]

        sentence = Sentence(test[comment_col].iloc[i], language_code='de')
        [sentence.add_label(label_type, l) for l in labels]
        comments_test.append(sentence)
    
    print('Created sentences.')

    train = SentenceDataset(comments_train)
    test = SentenceDataset(comments_test)

    corpus = Corpus(train=train, test=test)
    print('Made corpus.')
    
    tars.add_and_switch_to_new_task("target classification", 
                                    label_dictionary=corpus.make_label_dictionary(label_type = label_type),
                                    label_type=label_type)
    
    trainer = ModelTrainer(tars, corpus)
    trainer.train(base_path='few_shot/target', 
                  mini_batch_size=8, 
                  max_epochs=30, 
                  learning_rate=0.001,
                  save_final_model=True,
                  create_file_logs=True,
                  create_loss_file=True,
                  main_evaluation_metric = ("micro avg", "f1-score", "macro f1-score"),
                )


def check_results(path_res: str, data: pd.DataFrame):
    res = np.genfromtxt(path_res, delimiter=",", usemask=True, skip_header=0) > 0
    # true_full = data[data.columns[1:-4]].to_numpy()

    for i, col in enumerate(data.columns[1:-4]):
        pred = pd.DataFrame({col: res[:, i]})
        true = data[col]

        print(f"{col} accuracy:", accuracy_score(true, pred))
        print(f"Weighted {precision_recall_fscore_support(true, pred, average='weighted')}")
        print(f"Macro {precision_recall_fscore_support(true, pred, average='macro')}")
        print(f"Per class {precision_recall_fscore_support(true, pred, average=None)}\n\n")

        ConfusionMatrixDisplay.from_predictions(true, pred, normalize='all')
        plt.title(col)
        plt.savefig(f'plots_zero/{col}.png')


def train_few_binary(train, test1, test2, test3, comment_col, classes, label_col):
    tars = TARSClassifier.load('tars-base')
    
    label_type = 'target_class'
    
    comments_train = []
    for i in range(train.shape[0]):
        label = classes[0] if train[label_col].iloc[i] == 1 else classes[1]

        sentence = Sentence(train["comment_preprocessed_legacy"].iloc[i], language_code='de')
        sentence.add_label(label_type, label)
        comments_train.append(sentence)
    
    comments_test = []
    c_test1 = []
    for i in range(test1.shape[0]):
        label = classes[0] if test1[label_col].iloc[i] == 1 else classes[1]

        sentence = Sentence(test1["comment_preprocessed_legacy"].iloc[i], language_code='de')
        c_test1.append(test1["comment_preprocessed_legacy"].iloc[i])
        sentence.add_label(label_type, label)
        comments_test.append(sentence)

    comments_test2 = []
    c_test2 = []
    for i in range(test2.shape[0]):
        label = classes[0] if test2[label_col].iloc[i] == 1 else classes[1]

        sentence = Sentence(test2[comment_col].iloc[i], language_code='de')
        c_test2.append(test2[comment_col].iloc[i])
        sentence.add_label(label_type, label)
        comments_test2.append(sentence)
    
    comments_test3 = []
    c_test3 = []
    for i in range(test3.shape[0]):
        label = classes[0] if test3[label_col].iloc[i] == 1 else classes[1]

        sentence = Sentence(test3[comment_col].iloc[i], language_code='de')
        c_test3.append(test3[comment_col].iloc[i])
        sentence.add_label(label_type, label)
        comments_test3.append(sentence)
    
    print('Created sentences.')

    train = SentenceDataset(comments_train)
    test = SentenceDataset(comments_test)

    corpus = Corpus(train=train, test=test)
    corpus_eval = Corpus(train=SentenceDataset(comments_test2), test=SentenceDataset(comments_test3))
    print('Made corpus.')
    
    tars.add_and_switch_to_new_task("target classification", 
                                    label_dictionary=corpus.make_label_dictionary(label_type = label_type),
                                    label_type=label_type)
    
    trainer = ModelTrainer(tars, corpus)
    trainer.train(base_path=f'few_shot/target/{label_col}_minor', 
                  mini_batch_size=16, 
                  max_epochs=30, 
                  save_final_model=True,
                  create_file_logs=True,
                  create_loss_file=True,
                  learning_rate=0.001,
                  main_evaluation_metric = ("macro avg", "f1-score", "macro f1-score"),
                )
    
    print("Train eval:")
    res_eval = tars.evaluate(corpus.train, gold_label_type=label_type, mini_batch_size=1, out_path=f"few_shot/res/train_{label_col}_minor_predictions.txt")
    print(res_eval)
    
    print("Main eval:")
    res_eval = tars.evaluate(corpus.test, gold_label_type=label_type, mini_batch_size=1, out_path=f"few_shot/res/main_{label_col}_minor_predictions.txt")
    print(res_eval)

    print("Repr. eval:")
    res_expert = tars.evaluate(corpus_eval.train, gold_label_type=label_type, mini_batch_size=1, out_path=f"few_shot/res/repr_{label_col}_minor_predictions.txt")
    print(res_expert)

    print("Expert eval:")
    res_expert = tars.evaluate(corpus_eval.test, gold_label_type=label_type, mini_batch_size=1, out_path=f"few_shot/res/expert_{label_col}_minor_predictions.txt")
    print(res_expert)

path, path_test1, path_test2, path_test3 = "data/processed_training_main_v4.csv", "data/processed_evaluation_main_v4.csv", "data/processed_evaluation_representative_v4.csv", "data/processed_evaluation_expert_v4.csv"
comment_col = 'comment'
train = read_data(path)
train = train[train.targeted == 1]

test1 = read_data(path_test1)
test1 = test1[test1.targeted == 1]

test2 = read_data(path_test2)
test2 = test2[test2.targeted == 1]

test3 = read_data(path_test3)
test3 = test3[test3.targeted == 1]

print('Read files.')


train_few(train, test1, "comment_preprocessed_legacy")


# classes_ger = [
#         # "geschlecht", "alter", "sexualitat", "religion", "nationalitaet", 
#         # "behinderung", 
#         "sozialer status", "politische ansichten",  "aussehen", "andere"]

# classes_eng = [
#         # "gender", "age", "sexuality", "religion", "nationality", 
#         # "disability", 
#         "social_status", "political_views", "appearance", "other"]

# # for e, g in zip(classes_eng, classes_ger):
# #     classes_binary = [f"{g} hassrede", f"keine {g} hassrede"]
# #     train_few_binary(train, test1, test2, test3, comment_col, classes=classes_binary, label_col=e)


# for e, g in zip(classes_eng, classes_ger):
#     classes_binary = [f"{g} hassrede", f"keine {g} hassrede"]
    
#     # n = sum(train[e] == 1)
#     # positive = train[train[e] == 1]
#     # negative = train[train[e] == 0]
#     # negative = negative.sample(n=n)
#     # train_new = pd.concat([positive, negative], axis=0)
#     train_few_binary(train, test1, test2, test3, comment_col, classes=classes_binary, label_col=e)

