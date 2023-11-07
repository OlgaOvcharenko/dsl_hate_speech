
from flair.models import TARSClassifier
from flair.data import Sentence
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from flair.data import Corpus
from flair.datasets import SentenceDataset


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
        print(x)

        tars.predict_zero_shot(x, classes, multi_label=True)
        res[i, [classes.index(label.value) for label in x.labels]] = [label.score for label in x.labels]
        print(x)

        if i % 100 == 0:
            print(f"{i}th iteration of {data.shape[0]}")

    print(res)
    np.savetxt("data/zero_shot.csv", res, delimiter=",")

def check_results(path_res: str, data: pd.DataFrame):
    res = np.genfromtxt(path_res, delimiter=",", usemask=True, skip_header=0) > 0
    true_full = data[data.columns[1:-4]].to_numpy()

    print(res.shape)
    print(true_full.shape)

    # print(f"All accuracy:", accuracy_score(true_full, res))
    # print(f"Weighted {precision_recall_fscore_support(true_full, res, average='weighted')}")
    # print(f"Macro {precision_recall_fscore_support(true_full, res, average='macro')}")
    # print(f"Per class {precision_recall_fscore_support(true_full, res, average=None)}\n\n")

    # ConfusionMatrixDisplay.from_predictions(true_full, res, normalize='all')
    # plt.title("All")
    # plt.savefig(f'plots_zero/all.png')

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

# path = "data/processed_comments_train_v1.csv"
# comment_col = 'comment'
# data = read_data(path)
# data = data[data.targeted == 1]
# print('Read file.')
# train_zero(data, comment_col)
# check_results("data/zero_shot.csv", data.iloc[0:1000])

s = 'das sind doch alles grüne schläfer/v-frauenmänner. die volkspartei union hat kein problem'

comment = Sentence(s, language_code='de')
print('Created sentences.')

tars = TARSClassifier.load('tars-base')
classes = ["geschlecht", "alter", "sexualitat", "religion", "nationalitaet", 
        "behinderung", "sozialer status", "politische ansichten",  "aussehen"]

tars.predict_zero_shot(comment, classes, multi_label=False)
print(comment)

tars.predict_zero_shot(comment, classes, multi_label=True)
print(comment)
