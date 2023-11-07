
from flair.models import TARSClassifier
from flair.data import Sentence
import pandas as pd
import numpy as np


def read_data(path: str):
    return pd.read_csv(path, encoding='utf-8')

path = "data/processed_comments_train_v1.csv"
comment_col = 'comment'
data = read_data(path)
data = data[data.targeted == 1]
print('Read file.')

comments = [Sentence(s, language_code='de') for s in data[comment_col]]  # .iloc[0:2]
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

