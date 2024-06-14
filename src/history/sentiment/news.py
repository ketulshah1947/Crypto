import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob


def fetch_crypto_news():
    response = requests.get("https://cryptonews.com/news/")
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    for item in soup.find_all(
        "div", class_="posts__item"
    ):  # Adjust the class based on the website's HTML structure
        headline = item.find("h4", class_="posts__title").text.strip()
        timestamp = item.find("time", class_="posts__date")[
            "datetime"
        ]  # Assuming the datetime is in the 'datetime' attribute
        headlines.append([timestamp, headline])

    df = pd.DataFrame(headlines, columns=["timestamp", "headline"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


def process_historical_data():
    crypto_news_df = fetch_crypto_news()
    crypto_news_df.to_csv("data/crypto_news.csv", index=False)
    print(crypto_news_df.head())
    news_df = pd.read_csv("data/crypto_news.csv", parse_dates=["timestamp"])
    # Apply sentiment analysis
    news_df["sentiment"] = news_df["headline"].apply(get_sentiment)
    # Load historical price data
    price_df = pd.read_csv(
        "data/bitcoin_historical_data.csv", parse_dates=["timestamp"]
    )
    # Merge news sentiment with price data based on the nearest timestamp
    # Assuming news data is sparse compared to price data, we'll perform a forward fill
    combined_df = pd.merge_asof(
        price_df.sort_values("timestamp"),
        news_df.sort_values("timestamp"),
        on="timestamp",
    )
    combined_df["sentiment"] = combined_df["sentiment"].fillna(
        0
    )  # Fill NaN sentiment values with neutral sentiment
    # Save the combined dataframe
    combined_df.to_csv("data/bitcoin_combined_with_news_sentiment.csv", index=False)
    print(combined_df.head())
