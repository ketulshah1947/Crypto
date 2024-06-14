from typing import Dict, Optional

import pandas as pd
from ta import momentum, trend, volatility, volume

from src.history.finance.finance_history import process_finance_historical_data


class FinancialData:
    bitcoin_history: pd.DataFrame
    technical_indicators_features: pd.DataFrame
    bitcoin_time_features: pd.DataFrame

    def __init__(
        self,
        bitcoin_history: pd.DataFrame,
        technical_indicators_features: pd.DataFrame,
        bitcoin_time_features: pd.DataFrame,
    ):
        super().__init__()
        self.bitcoin_history = bitcoin_history
        self.technical_indicators_features = technical_indicators_features
        self.bitcoin_time_features = bitcoin_time_features


# We'll calculate technical indicators such as Moving Averages, RSI, MACD, Bollinger Bands, etc.
def _calculate_technical_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
    # dataframe = pd.read_csv(
    #     "data/bitcoin_historical_data.csv", parse_dates=["timestamp"]
    # )
    # Calculate technical indicators using the 'close' price
    dataframe["sma_50"] = trend.sma_indicator(dataframe["close"], window=50)
    dataframe["sma_200"] = trend.sma_indicator(dataframe["close"], window=200)
    dataframe["rsi"] = momentum.rsi(dataframe["close"], window=14)
    dataframe["macd"] = trend.macd(dataframe["close"])
    dataframe["bollinger_mavg"] = volatility.bollinger_mavg(dataframe["close"])
    dataframe["bollinger_hband"] = volatility.bollinger_hband(dataframe["close"])
    dataframe["bollinger_lband"] = volatility.bollinger_lband(dataframe["close"])
    dataframe["atr"] = volatility.average_true_range(
        dataframe["high"],
        dataframe["low"],
        dataframe["close"],
    )
    dataframe["obv"] = volume.on_balance_volume(dataframe["close"], dataframe["volume"])
    # Save the dataframe with technical indicators
    # technical_indicators_features.to_csv('data/bitcoin_technical_indicators.csv', index=False)
    return dataframe


# We'll extract time-based features such as the time of day, day of the week, and seasonal trends from the timestamp.
def _calculate_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    # bitcoin_time_features = pd.read_csv(
    #     "data/bitcoin_historical_data.csv", parse_dates=["timestamp"]
    # )
    # Extract time-based features
    dataframe["hour"] = dataframe["timestamp"].dt.hour
    dataframe["day_of_week"] = dataframe["timestamp"].dt.dayofweek
    dataframe["day_of_month"] = dataframe["timestamp"].dt.day
    dataframe["month"] = dataframe["timestamp"].dt.month
    # Save the dataframe with time features
    # dataframe.to_csv('data/bitcoin_time_features.csv', index=False)
    return dataframe


def analyse_financial_history() -> FinancialData:
    bitcoin_history = process_finance_historical_data()
    if not bitcoin_history:
        raise RuntimeError("No historical data")
    technical_indicators_features = _calculate_technical_indicators(bitcoin_history)
    bitcoin_time_features = _calculate_time_features(bitcoin_history)
    return FinancialData(
        bitcoin_history=bitcoin_history,
        technical_indicators_features=technical_indicators_features,
        bitcoin_time_features=bitcoin_time_features,
    )
