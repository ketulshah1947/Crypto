import ccxt
import numpy as np

from src.history.finance.finance_history import fetch_historical_data
from src.model import build_lstm_model
from src.preprocess import preprocess_data
from src.utils import create_dataset


def main():
    exchange_id = "binance"
    symbol = "BTC/USDT"
    timeframe = "1m"  # 1-minute data
    since = ccxt.binance().parse8601(
        "2023-06-11T00:00:00Z"
    )  # Adjust the date to one year ago from today

    print("Fetching historical data...")
    data = fetch_historical_data(exchange_id, symbol, timeframe, since)

    print("Preprocessing data...")
    scaled_data, scaler = preprocess_data(data)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Reshape data to fit LSTM input requirements
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print("Building LSTM model...")
    model = build_lstm_model(X.shape)

    print("Training model...")
    model.fit(X, y, batch_size=64, epochs=5)

    # For demonstration, we just print the model summary
    print(model.summary())


if __name__ == "__main__":
    main()
