"""
LSTM model for univariate time-series forecasting + simple training wrapper.

Used by app.inference._train_dl_model via:
    from . import model_lstm as mdl
    mdl.train_lstm(series, window=window, epochs=epochs)
"""

import gc
from typing import Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Default config (can be overridden from inference.py if needed)
DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON = 28


# ---------------------------------------------------------------------
# Helper: make sliding windows from a 1D series
# ---------------------------------------------------------------------
def _make_univariate_windows(
    series: Sequence[float],
    window: int,
    horizon: int,
):
    """Create (X, Y) for seq2seq forecasting from a 1D series."""
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

    X = np.array(X, dtype="float32")[..., np.newaxis]  # (samples, window, 1)
    Y = np.array(Y, dtype="float32")                   # (samples, horizon)
    return X, Y


# ---------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------
def build_lstm(
    input_shape=(DEFAULT_LOOKBACK, 1),
    horizon: int = DEFAULT_HORIZON,
) -> Model:
    """Build a moderately expressive LSTM network.

    The same builder is used both for:
      * local univariate models (Forecast page)
      * multivariate global models (training.fit_cv), where ``input_shape``
        has ``n_features > 1``.

    Architecture is chosen to balance capacity and CPU usage so it can
    still run comfortably on a laptop:
      * 2 stacked LSTM layers (64 → 32 units) with dropout
      * a small dense "head" before the horizon output
    """
    inputs = Input(shape=input_shape)

    # First LSTM layer keeps the full sequence
    x = LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    # Second LSTM layer returns only the last hidden state
    x = LSTM(32, return_sequences=False, dropout=0.2)(x)

    # Small dense head
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(horizon)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mae",
        metrics=["mse"],
    )
    return model


# ---------------------------------------------------------------------
# Training wrapper used by app.inference (univariate case)
# ---------------------------------------------------------------------
def train_lstm(
    series,
    window: int = DEFAULT_LOOKBACK,
    epochs: int = 10,
    horizon: int = DEFAULT_HORIZON,
    batch_size: int = 32,
) -> Model:
    """Train an LSTM on a univariate series.

    This wrapper is used by the Streamlit Forecast page via
    ``run_dl_forecast_with_uncertainty``. It uses a simple validation
    split and EarlyStopping to avoid overfitting.
    """
    tf.keras.backend.clear_session()

    X, Y = _make_univariate_windows(series, window=window, horizon=horizon)

    model = build_lstm(input_shape=(window, 1), horizon=horizon)

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
