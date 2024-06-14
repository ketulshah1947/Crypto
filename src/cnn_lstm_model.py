import pandas as pd
from keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam


class CnnLstmModel:
    model: Sequential
    predictions_dataframe: pd.DataFrame

    def __init__(self, model: Sequential, predictions_dataframe: pd.DataFrame):
        self.model = model
        self.predictions_dataframe = predictions_dataframe


def create_cnn_lstm_model(
    enhanced_df: pd.DataFrame, combined_df: pd.DataFrame
) -> CnnLstmModel:
    # Load enhanced features data
    # enhanced_df = pd.read_csv('data/bitcoin_enhanced_features.csv', parse_dates=['timestamp'])
    # combined_df = pd.read_csv('data/bitcoin_combined_features.csv', parse_dates=['timestamp'])
    # Prepare the data for training
    X = enhanced_df.drop(columns=["timestamp"]).values
    y = combined_df["close"].values[1:]  # Next closing price as the target
    X, y = X[:-1], y  # Align lengths
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    # Reshape the data for CNN-LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # Define the hybrid CNN-LSTM model
    model = Sequential()
    model.add(
        Conv1D(
            filters=64,
            kernel_size=2,
            activation="relu",
            input_shape=(X_train.shape[1], 1),
        )
    )
    model.add(Conv1D(filters=64, kernel_size=2, activation="relu"))
    model.add(LSTM(50, activation="relu", return_sequences=True))
    model.add(LSTM(50, activation="relu"))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss="mse")
    # Train the model
    history = model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
    )
    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Model Test Loss: {loss}")
    # Save the model in the recommended Keras format
    model.save("hybrid_cnn_lstm_model.keras")
    # Make predictions
    y_pred = model.predict(X_test)
    # Save predictions and actual values for comparison
    predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.flatten()})
    # predictions_df.to_csv('data/bitcoin_predictions.csv', index=False)
    # print(predictions_df.head())
    return CnnLstmModel(model=model, predictions_dataframe=predictions_df)
