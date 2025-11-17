import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

INPUT_FILE = "data/processed/sentiment_scored.csv"
OUTPUT_FILE = "data/processed/topics_lda.csv"
N_TOPICS = 5  # adjust number of topics

def run_topic_modeling():
    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("Reading:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)
    docs = df["cleaned"].tolist()

    # Vectorize text
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(docs)

    # Train LDA model
    lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42)
    lda_topics = lda.fit_transform(X)

    # Assign each document its dominant topic
    dominant_topics = lda_topics.argmax(axis=1)
    df["topic"] = dominant_topics

    # Save results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved topic file to: {OUTPUT_FILE}")

    # Print top words per topic
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]  # top 10 words
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
