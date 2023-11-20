import pandas as pd
from sklearn.metrics import f1_score
import math

golds_data = pd.read_csv('data/golds.csv', delimiter=',')

commentators = ["Hate Speech_KD", "Hate Speech_FG", "Hate Speech_SK"]
for c in commentators:
    idx = pd.notnull(golds_data[c])

    other = filter(lambda i: i!=c, commentators)
    y_true = ((golds_data.loc[idx])[other].sum(axis=1) / 2) >= 0.5

    # y_true = golds_data.loc[idx, 'Konsensus HS']
    y_pred = golds_data.loc[idx, c]

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"For {c} macro F1 is {round(f1, 3)}")
