import numpy as np
import pandas as pd
from keras import Model
from keras.src.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input


# Custom Callback for Training Progress
class TrainingProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']}")

    def on_batch_end(self, batch, logs=None):
        progress = (batch + 1) / self.params["steps"]
        print(
            f"\rBatch {batch + 1}/{self.params['steps']} - Progress: {progress:.2%}",
            end="",
        )

    def on_epoch_end(self, epoch, logs=None):
        print("\n")


def build_latent_features(combined_features: pd.DataFrame) -> pd.DataFrame:
    # Load combined features data
    # combined_df = pd.read_csv('../data/bitcoin_combined_features.csv', parse_dates=['timestamp'])
    # Drop timestamp column and scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_features.drop(columns=["timestamp"]))
    # Check for NaN values in scaled data
    if np.isnan(scaled_data).any():
        print("NaN values found in the scaled data. Cleaning up...")
        scaled_data = np.nan_to_num(scaled_data)
    # Verify scaled data
    print("Sample of scaled data:")
    print(f"scaled_data : {scaled_data[:5]}")
    # Define autoencoder architecture
    input_dim = scaled_data.shape[1]
    encoding_dim = 14  # Dimension of the latent space

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(input_layer)
    decoded = Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    # Train the autoencoder with progress callback
    history = autoencoder.fit(
        scaled_data,
        scaled_data,
        epochs=50,
        batch_size=32,
        shuffle=True,
        callbacks=[TrainingProgressCallback()],
    )

    print("Autoencoder training loss:", history.history["loss"][-1])
    # Extract the encoder part of the autoencoder
    encoder = Model(input_layer, encoded)
    # Get the latent representations
    latent_representations = encoder.predict(scaled_data)
    # Verify latent representations
    print("Shape of latent representations:", latent_representations.shape)
    print("Sample of latent representations:")
    print(latent_representations[:5])
    # Create a dataframe with the latent representations
    latent_df = pd.DataFrame(
        latent_representations, columns=[f"latent_{i + 1}" for i in range(encoding_dim)]
    )
    latent_df["timestamp"] = combined_features["timestamp"]
    # Save the latent representations
    # latent_df.to_csv('data/bitcoin_latent_features.csv', index=False)
    # print(latent_df.head())
    return latent_df
