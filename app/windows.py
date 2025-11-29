"""
Window generation utilities for deep learning time-series forecasting.

This module converts a feature-engineered dataframe into overlapping
input/output windows suitable for LSTM, TCN, and Transformer models.

Given:
    LOOKBACK = number of past days (e.g., 56)
    HORIZON  = forecast horizon (e.g., 28)

It builds:
    - X : (num_samples, lookback, num_features)
    - Y : (num_samples, horizon)
    - T : timestamps marking the end of each input window
"""

import numpy as np
import pandas as pd


def make_windows(df: pd.DataFrame,
                 lookback: int,
                 horizon: int,
                 feature_cols: list,
                 target_col: str):
    """
    Slice a time-series dataframe into supervised learning windows.

    Parameters
    ----------
    df : pd.DataFrame
        Must include:
        - feature_cols   (numeric & boolean features)
        - target_col     (e.g., 'sales')
        - 'date' column  (for timestamp tracking)
    lookback : int
        Number of historical days per input window.
    horizon : int
        Number of days to predict ahead.
    feature_cols : list
        Columns used as model inputs.
    target_col : str
        Column to forecast.

    Returns
    -------
    X : np.ndarray
        Shape: (samples, lookback, num_features)
    Y : np.ndarray
        Shape: (samples, horizon)
    T : np.ndarray
        Shape: (samples,) timestamps marking window end dates
    """

    # Ensure sorted by date
    df = df.sort_values("date")

    # Extract needed arrays
    ts_data = df[feature_cols].values.astype("float32")
    target_data = df[target_col].values.astype("float32")
    timestamps = df["date"].values

    X, Y, T = [], [], []

    max_idx = len(df) - lookback - horizon + 1

    for i in range(max_idx):
        # Input window
        x_window = ts_data[i : i + lookback]
        X.append(x_window)

        # Output horizon
        y_window = target_data[i + lookback : i + lookback + horizon]
        Y.append(y_window)

        # Timestamp for the end of the input window
        T.append(timestamps[i + lookback - 1])

    return (
        np.array(X, dtype="float32"),
        np.array(Y, dtype="float32"),
        np.array(T),
    )
