import pandas as pd
from langdetect import detect

def read_data_to_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def detect_language(df: pd.DataFrame, comments_col: str = "kommentar_original", lang_col: str = "lang") -> pd.DataFrame:
    langs = []
    for val in  df[comments_col]:
        try:
            language = detect(val)
        except:
            language = "err"
        langs.append(language)
    df[lang_col] = langs
    print(set(langs))
    return df


df = read_data_to_df("data/all_DEFR_comments_27062022.csv")
df = detect_language(df)
df.to_csv("data/all_comments_lang.csv", index=False)
