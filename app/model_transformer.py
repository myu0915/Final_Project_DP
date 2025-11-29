"""
Tiny Transformer encoder for univariate time-series forecasting
+ simple training wrapper.

Used by app.inference._train_dl_model via:
    from . import model_transformer as mdl
    mdl.train_transformer(series, window=window, epochs=epochs)
"""

import gc
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Embedding,
    Add,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
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


def build_transformer(
    input_shape=(DEFAULT_LOOKBACK, 1),
    horizon: int = DEFAULT_HORIZON,
    num_heads: int = 2,
    ff_dim: int = 64,
) -> Model:
    """
    Tiny Transformer encoder:
      - positional embedding
      - 1 MultiHeadAttention + FFN block
      - global average pooling
      - dense head → horizon outputs
    """
    lookback = input_shape[0]

    inputs = Input(shape=input_shape)          # (T, 1)

    # Positional encoding (learned)
    positions = tf.range(start=0, limit=lookback, delta=1)
    pos_emb = Embedding(input_dim=lookback, output_dim=1)(positions)  # (T, 1)

    x = inputs + pos_emb  # broadcast add

    # Self-attention block
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=ff_dim,
    )(x, x)

    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward block
    ffn = tf.keras.Sequential(
        [
            Dense(ff_dim, activation="relu"),
            Dense(1),
        ]
    )(x)

    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Pool over time
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


def train_transformer(
    series,
    window: int = DEFAULT_LOOKBACK,
    epochs: int = 10,
    horizon: int = DEFAULT_HORIZON,
    batch_size: int = 32,
) -> Model:
    """
    Train the tiny Transformer on a univariate series.
    """
    tf.keras.backend.clear_session()

    X, Y = _make_univariate_windows(series, window=window, horizon=horizon)
    model = build_transformer(input_shape=(window, 1), horizon=horizon)

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
