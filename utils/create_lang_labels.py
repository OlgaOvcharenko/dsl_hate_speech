import pandas as pd
from langdetect import detect

def read_data_to_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def detect_language(df: pd.DataFrame, comments_col: str = "kommentar", lang_col: str = "lang") -> pd.DataFrame:
    return pd.concat([df, pd.get_dummies([detect(str(val)) if isinstance(val, str)  and val.isalpha() else str(val) for val in  df[comments_col]])], axis=1)


df = read_data_to_df("data/all_DEFR_comments_27062022.csv")
df = detect_language(df)
df.to_csv("data/all_comments_lang.csv", index=False)
