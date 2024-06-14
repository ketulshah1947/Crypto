# Apply the monkey patch
import os

# os.environ['NUMPY_DTYPE_USE_NEW_OBJECT'] = '1'

import backtrader as bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.training.predictive_strategy import PredictiveStrategy


# Custom data feed class
class CustomPandasData(bt.feeds.PandasData):
    lines = ('predicted',)
    params = (('predicted', -1),)

    # Define the columns
    datafields = bt.feeds.PandasData.datafields + ['predicted']


def perform_backtesting(predictions_df: pd.DataFrame, enhanced_df: pd.DataFrame, combined_df: pd.DataFrame):
    try:
        # Load the predictions and actual values
        predictions_df = pd.read_csv('data/bitcoin_predictions.csv')
        enhanced_df = pd.read_csv('data/bitcoin_enhanced_features.csv', parse_dates=['timestamp'])
        enhanced_df = enhanced_df.dropna()
        combined_df = pd.read_csv('data/bitcoin_combined_features.csv', parse_dates=['timestamp'])
        combined_df = combined_df.dropna()

        # Merge predictions with timestamps
        predictions_df['timestamp'] = enhanced_df['timestamp']
        predictions_df.set_index('timestamp', inplace=True)

        # Ensure the dataframe for backtrader has necessary columns
        combined_df.set_index('timestamp', inplace=True)
        predictions_df['close'] = combined_df['close'].shift(-1)  # Next closing price as the target
        predictions_df = predictions_df.dropna()  # Drop rows with NaN values

        # Filter out rows with non-finite values
        predictions_df = predictions_df[np.isfinite(predictions_df).all(1)]

        # Ensure data types are correct
        predictions_df['close'] = predictions_df['close'].astype(float)
        predictions_df['Predicted'] = predictions_df['Predicted'].astype(float)

        # Prepare the data for backtrader
        data = CustomPandasData(dataname=predictions_df)

        # Initialize the backtrader engine
        cerebro = bt.Cerebro()
        cerebro.addstrategy(PredictiveStrategy, predictions=predictions_df['Predicted'].values)

        # Add data to the engine
        cerebro.adddata(data, name="ok")

        # Set the initial cash
        cerebro.broker.set_cash(100000)

        # Set the commission
        cerebro.broker.setcommission(commission=0.001)

        # Run the backtest
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        cerebro.run()
        print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Plot the results
        cerebro.plot(style='candlestick')
    except AttributeError as e:
        print(f"AttributeError encountered: {e}")
    except Exception as e:
        print(f"An error encountered: {e}")
