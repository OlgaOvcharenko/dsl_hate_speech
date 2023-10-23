import pandas as pd
from langdetect import detect

def read_data_to_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def detect_language(df: pd.DataFrame, comments_col: str = "kommentar_original", lang_col: str = "lang") -> pd.DataFrame:
    # return pd.concat([df, pd.get_dummies([detect(str(val)) if isinstance(val, str)  and val.isalpha() else str(val) for val in  df[comments_col]])], axis=1)
    # return [detect(str(val)) if isinstance(val, str) and val.isalpha() else str('de') for val in  df[comments_col]]
    # return [detect(str(val)) if str(val) else 't' for val in  df[comments_col]]
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
