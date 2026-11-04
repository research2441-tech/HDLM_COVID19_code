"""Keras definitions of the two HDLM stages and the neural baselines."""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_record_cnn(input_shape=(224, 224, 3), dropout=0.25):
    """Stage-1 CNN of Table 1: four Conv2D, three MaxPooling2D, four Dropout, two Dense.

    The layer widths reproduce the parameter counts reported in Table 1
    (5,723,457 trainable parameters).
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inp)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="record_cnn")


def build_hybrid(window, n_feat, n_out, filters=32, kernel=3, units=(64, 32), dropout=0.2):
    """Stage-2 forecaster: 1-D convolutional encoder feeding a two-level stacked LSTM."""
    inp = layers.Input(shape=(window, n_feat))
    x = layers.Conv1D(filters, kernel, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(filters, kernel, padding="causal", activation="relu")(x)
    x = layers.LSTM(units[0], return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units[1])(x)
    out = layers.Dense(n_out)(x)
    return models.Model(inp, out, name="cnn_stacked_lstm")


def build_lstm(window, n_feat, n_out, units=64):
    inp = layers.Input(shape=(window, n_feat))
    x = layers.LSTM(units)(inp)
    return models.Model(inp, layers.Dense(n_out)(x), name="lstm")


def build_stacked_lstm(window, n_feat, n_out, units=(64, 32), dropout=0.2):
    inp = layers.Input(shape=(window, n_feat))
    x = layers.LSTM(units[0], return_sequences=True)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units[1])(x)
    return models.Model(inp, layers.Dense(n_out)(x), name="stacked_lstm")


def build_cnn1d(window, n_feat, n_out, filters=32, kernel=3):
    inp = layers.Input(shape=(window, n_feat))
    x = layers.Conv1D(filters, kernel, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(filters, kernel, padding="causal", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    return models.Model(inp, layers.Dense(n_out)(x), name="cnn1d")


BUILDERS = {
    "CNN-1D": build_cnn1d,
    "LSTM": build_lstm,
    "Stacked LSTM": build_stacked_lstm,
    "HDLM (CNN + stacked LSTM)": build_hybrid,
}


def count_params(model):
    trainable = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    frozen = int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
    return trainable, frozen
