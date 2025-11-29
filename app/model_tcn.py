"""
TCN (Temporal Convolutional Network) for univariate forecasting
+ simple training wrapper.

Used by app.inference._train_dl_model via:
    from . import model_tcn as mdl
    mdl.train_tcn(series, window=window, epochs=epochs)
"""

import gc
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON = 28


def _make_univariate_windows(
    series: Sequence[float],
    window: int,
    horizon: int,
):
    arr = np.asarray(series, dtype="float32")
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    n_samples = len(arr) - window - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Series too short for window={window}, horizon={horizon}. "
            f"Got length={len(arr)}."
        )

    X, Y = [], []
    for i in range(n_samples):
        X.append(arr[i : i + window])
        Y.append(arr[i + window : i + window + horizon])

    X = np.array(X, dtype="float32")[..., np.newaxis]
    Y = np.array(Y, dtype="float32")
    return X, Y


def build_tcn(
    input_shape=(DEFAULT_LOOKBACK, 1),
    horizon: int = DEFAULT_HORIZON,
) -> Model:
    """
    Simple causal TCN with 3 dilated Conv1D blocks.
    """
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=3, padding="causal", dilation_rate=1)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(64, kernel_size=3, padding="causal", dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(64, kernel_size=3, padding="causal", dilation_rate=4)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)

    outputs = Dense(horizon)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mse"],
    )
    return model


def train_tcn(
    series,
    window: int = DEFAULT_LOOKBACK,
    epochs: int = 10,
    horizon: int = DEFAULT_HORIZON,
    batch_size: int = 32,
) -> Model:
    """
    Train a TCN on a univariate series.
    """
    tf.keras.backend.clear_session()

    X, Y = _make_univariate_windows(series, window=window, horizon=horizon)
    model = build_tcn(input_shape=(window, 1), horizon=horizon)

    es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model.fit(
        X,
        Y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    gc.collect()
    return model
