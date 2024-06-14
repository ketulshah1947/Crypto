import pandas as pd

from src.history.finance.analysis import FinancialData, analyse_financial_history
from src.history.sentiment.analysis import perform_sentiment_analysis


def analyse_history() -> pd.DataFrame:
    financialData: FinancialData = analyse_financial_history()
    reddit_sentiment_df = perform_sentiment_analysis()
    # Load data
    historical_df = financialData.bitcoin_history
    # historical_df = pd.read_csv(
    #     "data/bitcoin_historical_data.csv", parse_dates=["timestamp"]
    # )
    indicators_df = financialData.technical_indicators_features
    # indicators_df = pd.read_csv(
    #     "data/bitcoin_technical_indicators.csv", parse_dates=["timestamp"]
    # )
    time_features_df = financialData.bitcoin_time_features
    # time_features_df = pd.read_csv('bitcoin_time_features.csv', parse_dates=['timestamp'])
    # reddit_sentiment_df = pd.read_csv('data/bitcoin_reddit_sentiment.csv', parse_dates=['timestamp'])
    # news_df = pd.read_csv('data/crypto_news.csv', parse_dates=['timestamp'])
    # Merge historical and technical indicators data
    combined_df = historical_df.merge(
        indicators_df, on="timestamp", suffixes=("", "_indicators")
    )
    # Merge Reddit sentiment data
    combined_df = combined_df.merge(
        reddit_sentiment_df[["timestamp", "sentiment"]], on="timestamp", how="left"
    )
    combined_df = combined_df.merge(
        time_features_df, on="timestamp", suffixes=("", "_time")
    )
    combined_df["sentiment"] = combined_df["sentiment"].fillna(
        0
    )  # Fill NaN sentiment values with neutral sentiment
    # Merge news data using merge_asof to align with nearest timestamp
    # combined_df.to_csv('data/bitcoin_combined_features.csv', index=False)
    return combined_df
