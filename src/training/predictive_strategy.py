import backtrader as bt
import numpy as np


# Define a simple backtesting strategy
class PredictiveStrategy(bt.Strategy):
    params = (('predictions', None),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.predictions = self.params.predictions
        self.pred_index = 0

    def next(self):
        if self.order:
            return

        if self.pred_index < len(self.predictions):
            predicted_price = self.predictions[self.pred_index]
            current_price = self.dataclose[0]

            # Debugging: Print current and predicted prices
            print(f"Index: {self.pred_index}, Current price: {current_price}, Predicted price: {predicted_price}")

            # Check if the current or predicted price is NaN
            if np.isnan(current_price) or np.isnan(predicted_price):
                print(
                    f"NaN detected at index {self.pred_index}: Current price: {current_price}, Predicted price: {predicted_price}")

            if predicted_price > current_price and not np.isnan(current_price) and not np.isnan(predicted_price):
                self.order = self.buy()
            elif predicted_price < current_price and not np.isnan(current_price) and not np.isnan(predicted_price):
                self.order = self.sell()

            self.pred_index += 1

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED, Price: {order.executed.price}')
            elif order.issell():
                print(f'SELL EXECUTED, Price: {order.executed.price}')
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f'Order {order.ref} Canceled/Margin/Rejected: {order.info}')

            # Debugging: Print order details
        print(f"Order: {order}, Status: {order.getstatusname()}, Executed Price: {order.executed.price}")
