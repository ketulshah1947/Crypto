import pandas as pd
from textblob import TextBlob

from src.history.sentiment.reddit_history import process_reddit_posts


# Function to get sentiment polarity
def get_sentiment(text):
    if pd.isna(text):
        return 0  # Assign neutral sentiment for NaN text
    return TextBlob(text).sentiment.polarity


# We'll use NLP libraries like spaCy and TextBlob to perform sentiment analysis on the collected Reddit posts.
def perform_sentiment_analysis() -> pd.DataFrame:
    reddit_df = process_reddit_posts()
    # Load Reddit posts data
    # reddit_df = pd.read_csv('data/bitcoin_reddit_posts.csv', parse_dates=['timestamp'])
    # Apply sentiment analysis
    reddit_df["sentiment"] = reddit_df["text"].apply(get_sentiment)
    # Save the dataframe with sentiment scores
    # reddit_df.to_csv('data/bitcoin_reddit_sentiment.csv', index=False)
    return reddit_df
