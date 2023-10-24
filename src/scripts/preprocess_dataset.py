# %%
import polars as pl
import polars.selectors as cs
from polars import col as c

# %%
df = pl.read_csv(
    "../../data/all_comments_lang.csv", dtypes={"ArticleID": pl.Utf8, "ID": pl.Utf8}
)

# %%
df = (
    df.drop_nulls()  # TODO Ask about the NULL values
    .unique()  # TODO Ask about the duplicate rows
    .drop("ArticleID", "ID", "kommentar_original", "toxische_sprache")
    .filter(c("lang") != "fr")  # Remove the french comments
    .rename(
        {
            "kommentar": "comment",
            "geschlecht": "gender",
            "alter": "age",
            "sexualitaet": "sexuality",
            "nationalitaet": "nationality",
            "beeintraechtigung": "disability",
            "sozialer_status": "social_status",
            "politik": "political_views",
            "aussehen": "appearance",
            "andere": "other",
        }
    )  # Standardize the column names
    .rename({"label": "toxic"})  # Fix the toxicity labels
    .with_columns(
        targeted=pl.sum_horizontal(cs.all().exclude(["toxic", "comment", "lang"]) > 0)
    )  # Create a new column to indicate if the toxicity is targeted (# targets > 0)
    .filter(
        (c("targeted") == 0) | (c("toxic") == 1)
    )  # TODO Ask about these illogical (targeted but non-toxic) rows
    .cast(
        {cs.numeric(): pl.Int64, cs.boolean(): pl.Int64}
    )  # Cast the boolean variables to long
    .with_row_count("id")  # Add a column with the row id
)

# %%


def normalize_text(comment):
    return (
        comment.replace("ü", "ue")
        .replace("ö", "oe")
        .replace("ä", "ae")
        .replace("Ü", "UE")
        .replace("Ö", "OE")
        .replace("Ä", "AE")
    )


df = df.with_columns(comment=c("comment").map_elements(normalize_text))


# %%
df.write_csv("../../data/clean_comments_non-fr_v1.csv")

# %%
