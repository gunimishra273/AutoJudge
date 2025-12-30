import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_preprocess(csv_path="data/problems.csv"):
    df = pd.read_csv(csv_path)

    text_cols = [
        "title",
        "description",
        "input_description",
        "output_description"
    ]

    for col in text_cols:
        df[col] = df[col].apply(clean_text)

    df["combined_text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["input_description"] + " " +
        df["output_description"]
    )

    return df
