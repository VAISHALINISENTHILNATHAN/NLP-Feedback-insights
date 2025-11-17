import pandas as pd
INPUT_FILE = "data/processed/topics_merged.csv"

def run_insights():
    df=pd.read_casv(INPUT_FILE)
    insights={
    "sentiment distribution":df["sentiment"].value_counts().to_dict(),
    "top_negative_topics":df[df["sentiment"]=="NEGATIVE"]["topic"].value_counts().head(5).to_dict()
    }

    print("Insights generated successfully!")
    return insights
