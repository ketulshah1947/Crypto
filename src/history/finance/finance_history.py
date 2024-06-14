from typing import Optional

import pandas as pd
import requests


def _fetch_historical_data(
    symbol, interval, start_str, end_str=None
) -> Optional[pd.DataFrame]:
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_str,
        "endTime": end_str,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        df = pd.DataFrame(data, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)
        return df
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def process_finance_historical_data() -> Optional[pd.DataFrame]:
    symbol = "BTCUSDT"
    interval = "1h"  # 1 hour interval
    start_str = "1609459200000"  # Unix timestamp for 2021-01-01 00:00:00
    bitcoin_history = _fetch_historical_data(symbol, interval, start_str)
    # bitcoin_history.to_csv('data/bitcoin_historical_data.csv')
    return bitcoin_history
