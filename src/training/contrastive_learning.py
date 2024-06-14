import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


# Create positive and negative pairs
def create_pairs(data, labels):
    pairs = []
    labels = []
    n = data.shape[0]

    for i in range(n):
        # Create a positive pair
        pos_pair = [data[i], data[(i + 1) % n]]
        pairs.append(pos_pair)
        labels.append(1)
        # Create a negative pair
        neg_pair = [data[i], data[(i + np.random.randint(1, n)) % n]]
        pairs.append(neg_pair)
        labels.append(0)
    return np.array(pairs), np.array(labels)


# Define the Siamese network architecture
def build_siamese_model(input_shape):
    input = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation="relu")(input)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    model = Model(input, x)
    return model


# Define the contrastive loss function
def contrastive_loss(y_true, y_pred, margin=1):
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (1 - y_pred), 0))
    return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


def train_contrastive_model(latent_features: pd.DataFrame) -> pd.DataFrame:
    # Load latent features data
    # latent_features = pd.read_csv('data/bitcoin_latent_features.csv', parse_dates=['timestamp'])
    latent_data = latent_features.drop(columns=["timestamp"]).values
    # Generate pairs and labels
    pairs, pair_labels = create_pairs(latent_data, latent_data)
    input_shape = latent_data.shape[1:]
    siamese_model = build_siamese_model(input_shape)
    # Build the full model
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    processed_a = siamese_model(input_a)
    processed_b = siamese_model(input_b)
    distance = layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]))(
        [processed_a, processed_b]
    )
    output = layers.Dense(1, activation="sigmoid")(distance)
    model = Model([input_a, input_b], output)
    model.compile(optimizer=Adam(0.001), loss=contrastive_loss, metrics=["accuracy"])
    # Train the model
    model.fit([pairs[:, 0], pairs[:, 1]], pair_labels, batch_size=32, epochs=20)
    # Save the enhanced representations
    enhanced_representations = siamese_model.predict(latent_data)
    enhanced_df = pd.DataFrame(
        enhanced_representations,
        columns=[f"enhanced_{i + 1}" for i in range(enhanced_representations.shape[1])],
    )
    enhanced_df["timestamp"] = latent_features["timestamp"]
    # enhanced_df.to_csv("data/bitcoin_enhanced_features.csv", index=False)
    # print(enhanced_df.head())
    return enhanced_df
