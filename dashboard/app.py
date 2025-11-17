import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/processed/topics_merged.csv")

st.set_page_config(page_title="Customer Insights",layout="wide")
st.title("📊 Customer Feedback Insights Platform")

tab1,tab2,tab3,tab4=st.tabs(["Overview","Topic Explorer","Sentiment Trends","Search Reviews"])

with tab1:
    st.header("Overall Sentiment Distribution")
    fig=px.pie(df,names="sentiment")
    st.plotly_chart(fig)
    st.header("Top Negative Topics")
    neg=df[df["sentiment"]=="NEGATIVE"]["topic"].value_counts().head(10)
    st.bar_chart(neg)

with tab2:
    topic_id=st.selectbox("Choose Topic",df["topic"].unique())
    st.write(df[df["topic"]==topic_id][["review","sentiment"]].head(20))

with tab3:
    st.header("Sentiment over Time")
    if "date" in df.columns:
        trend=df.groupby("date")["sentiment"].value_counts().unstack().fillna(0)
        st.line_chart(trend)
    else:
        st.info("No date column available in dataset.")

with tab4:
    search=st.text_input("Search reviews:")
    if search:
        results=df[df["review"].str.contains(search,case=False)]
        st.write(results)
                         
                         
