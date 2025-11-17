import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = " ".join([lemm.lemmatize(w) for w in text.split() if w not in stop_words])
    return text

def preprocess_file(
    input_path="data/raw/reviews.csv",
    output_path="data/processed/cleaned_reviews.csv"
):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Reading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    if "review" not in df.columns:
        raise ValueError("Input CSV must contain a column named 'review'.")

    df["cleaned"] = df["review"].apply(clean_text)
    df.to_csv(output_path, index=False)

    print(f"Cleaned file saved to: {output_path}")
