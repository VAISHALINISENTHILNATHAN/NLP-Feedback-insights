from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os

INPUT_FILE = "data/processed/cleaned_reviews.csv"
OUTPUT_FILE = "data/processed/sentiment_scored.csv"

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_sentiment(text):
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    output = model(**tokens)
    scores = torch.nn.functional.softmax(output.logits, dim=1)
    label = torch.argmax(scores).item()
    confidence = scores[0][label].item()
    return model.config.id2label[label], confidence

def run_sentiment():
    print("Reading:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    sentiments = df["cleaned"].apply(get_sentiment)
    df["sentiment"] = sentiments.apply(lambda x: x[0])
    df["confidence"] = sentiments.apply(lambda x: x[1])

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Saved sentiment-scored file to:", OUTPUT_FILE)
