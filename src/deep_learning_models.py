import tensorflow as tf
from tensorflow import keras


def build_mlp(input_dim, learning_rate=0.001):
    """
    Common MLP architecture for both baseline and enhanced experiments.
    Only the number of input features differs.
    """

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),

        keras.layers.Dense(
            64,
            activation="relu"
        ),

        keras.layers.BatchNormalization(),

        keras.layers.Dropout(0.30),

        keras.layers.Dense(
            32,
            activation="relu"
        ),

        keras.layers.Dropout(0.20),

        keras.layers.Dense(
            5,
            activation="softmax"
        )
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_baseline_ann(input_dim, learning_rate=0.001):
    return build_mlp(
        input_dim=input_dim,
        learning_rate=learning_rate
    )


def build_enhanced_ann(input_dim, learning_rate=0.001):
    return build_mlp(
        input_dim=input_dim,
        learning_rate=learning_rate
    )

def build_lstm(input_shape, learning_rate=0.001):
    """
    Common LSTM architecture for baseline and enhanced experiments.

    input_shape:
        (sequence_length, number_of_features)
    """

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),

        keras.layers.LSTM(
            64,
            activation="tanh",
            return_sequences=False
        ),

        keras.layers.Dropout(0.30),

        keras.layers.Dense(
            32,
            activation="relu"
        ),

        keras.layers.Dropout(0.20),

        keras.layers.Dense(
            5,
            activation="softmax"
        )
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_gru(input_shape, learning_rate=0.001):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),

        keras.layers.GRU(
            64,
            activation="tanh",
            return_sequences=False
        ),

        keras.layers.Dropout(0.30),

        keras.layers.Dense(
            32,
            activation="relu"
        ),

        keras.layers.Dropout(0.20),

        keras.layers.Dense(
            5,
            activation="softmax"
        )
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model