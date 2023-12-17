import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import math


df_pred = pd.DataFrame(0, index=np.arange(5801), columns=["gender", "age", "sexuality", "religion", "nationality", "disability", "social_status", "political_views", "appearance", "other"]) 

df_eval = pd.read_csv("data/processed_evaluation_main_v4.csv")
df_eval = df_eval[df_eval["targeted"]==1]

f = open("llama_zero_shot.out", "r")
raw = f.read()

split_raw = [val.split("Decoded")[1] for val in raw.split("Generate prompt")[1:]]

target_categories = ["gender", "age", "sexuality", "religion", "nationality", "disability", "social status", "political views", "appearance", "other"]
target_cols = ["gender", "age", "sexuality", "religion", "nationality", "disability", "social_status", "political_views", "appearance", "other"]

for i in range(df_eval.shape[0]):
    text = df_eval["comment_preprocessed_legacy"].iloc[i]
    
    print(split_raw[i].replace("\n", "").split("OUTPUT:"))
    out_raw = split_raw[i].replace("\n", "").split("OUTPUT:")[1] if 

    for j in range(len(target_categories)):
        if target_categories[j] in out_raw:
            df_pred[target_cols[j]].iloc[i] = 1

macro, binary = [], []
print(df_pred.sum(axis=0))
for cat in [
    "nationality", "religion", "gender", "political_views", "sexuality", "age",
    "social_status", "appearance", "disability", "other"
]:
    true, pred = df_eval[cat], df_pred[cat]
    
    macro_f1 = precision_recall_fscore_support(true, pred, average='macro')[2]
    binary_f1 = precision_recall_fscore_support(true, pred, average='binary')[2]
    macro.append(round(macro_f1, 3))
    binary.append(round(binary_f1, 3))
    print(cat, macro_f1, binary_f1)

print(macro)
print(binary)