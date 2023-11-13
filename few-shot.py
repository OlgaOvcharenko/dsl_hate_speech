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
    
    label_type = 'target_class'
    
    comments_train = []
    for i in range(train.shape[0]):
        labels = [classes[j-1] for j in range(1, len(classes)+1) if train.iloc[i, j]]

        sentence = Sentence(train[comment_col].iloc[i], language_code='de')
        [sentence.add_label(label_type, l) for l in labels]
        comments_train.append(sentence)
    
    comments_test = []
    for i in range(test.shape[0]):
        labels = [classes[j-1] for j in range(1, len(classes)+1) if test.iloc[i, j]]

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
                  learning_rate=0.02, 
                  mini_batch_size=16, 
                  max_epochs=15, 
                  save_final_model=True,
                  create_file_logs=True,
                  create_loss_file=True
                )
    
    # Load few-shot TARS model
    tars = TARSClassifier.load('few_shot/target/final-model.pt')

    res = np.zeros((len(comments_test), len(classes)))

    for i, x in enumerate(comments_test):
        tars.predict(x, classes, multi_label=False)
        res[i, [classes.index(label.value) for label in x.labels]] = [label.score for label in x.labels]

        tars.predict(x, classes, multi_label=True)
        res[i, [classes.index(label.value) for label in x.labels]] = [label.score for label in x.labels]

        if i % 100 == 0:
            print(f"{i}th iteration of {test.shape[0]}")

    print(res)
    np.savetxt("data/zero_shot.csv", res, delimiter=",")


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


path, path_test = "data/processed_comments_train_v1.csv", "data/processed_comments_evaluation_v1.csv"
comment_col = 'comment'
train = read_data(path)
train = train[train.targeted == 1]

test = read_data(path).iloc[0:10]
test = test[test.targeted == 1].iloc[0:10]
print('Read file.')

train_few(train, test, comment_col)
# check_results("data/zero_shot.csv", test.iloc[0:1000])
