"""
Evaluation utilities for retail demand forecasting.

Contains:
- Metrics: MAPE, RMSE, MASE
- Naive(Last Value) baseline evaluation
- Simple ARIMA baseline evaluation helper
- Leaderboard builder
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None


# =====================================================================
# METRICS
# =====================================================================

def safe_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE) in a safe way.

    MAPE ≈ average of |(true - pred) / true| * 100%

    If y_true contains zeros, we add a small epsilon to prevent
    division by zero.
    """
    y_true = np.asarray(y_true, dtype="float32")
    y_pred = np.asarray(y_pred, dtype="float32")

    epsilon = 1e-6
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100.0


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE).

    - Penalizes large errors heavily
    - Same unit as target (e.g., unit sales)
    """
    y_true = np.asarray(y_true, dtype="float32")
    y_pred = np.asarray(y_pred, dtype="float32")
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mase(y_true, y_pred, y_train_history):
    """
    Mean Absolute Scaled Error (MASE).

    Compares model MAE on the forecast window to the MAE of
    a Naive(1) forecast on the training history.

    - MASE ≈ 1: model similar to naive
    - MASE < 1: model better than naive (good)
    - MASE > 1: model worse than naive (bad)
    """
    y_true = np.asarray(y_true, dtype="float32")
    y_pred = np.asarray(y_pred, dtype="float32")
    y_train_history = np.asarray(y_train_history, dtype="float32")

    # naive(1) scale: |y_t - y_{t-1}|
    q = np.mean(np.abs(y_train_history[1:] - y_train_history[:-1]))

    if q < 1e-6:
        # flat series → fall back to MAPE
        return safe_mape(y_true, y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    return mae / q


# =====================================================================
# NAIVE BASELINE — "LAST VALUE" FORECAST
# =====================================================================

def eval_naive_last_value(X_windowed, Y_true, y_train_history,
                          lag_1_idx: int, horizon: int):
    """
    Evaluate a simple Naive(Last Value) baseline:

    For each window:
        - Take 'sales_lag_1' from the last row of the input window
        - Forecast that value for every day in the horizon

    Parameters
    ----------
    X_windowed : np.ndarray
        Shape: (samples, lookback, num_features).
    Y_true : np.ndarray
        Shape: (samples, horizon).
    y_train_history : np.ndarray
        1D array of historical sales used to compute MASE scaling.
    lag_1_idx : int
        Index of the 'sales_lag_1' feature within the feature dimension.
    horizon : int
        Number of forecast days.

    Returns
    -------
    metrics : dict
        {
          "MAPE": ...,
          "RMSE": ...,
          "MASE": ...
        }
    """
    # last row of each window → yesterday's sales
    y_last = X_windowed[:, -1, lag_1_idx]  # shape: (samples,)

    # repeat across horizon
    y_pred = np.tile(y_last.reshape(-1, 1), (1, horizon))

    mape_val = safe_mape(Y_true, y_pred)
    rmse_val = rmse(Y_true, y_pred)
    mase_val = mase(Y_true, y_pred, y_train_history)

    return {
        "MAPE": mape_val,
        "RMSE": rmse_val,
        "MASE": mase_val,
    }


# =====================================================================
# ARIMA BASELINE HELPER
# =====================================================================

def eval_arima_baseline(train_series, y_true_window, order=(5, 1, 0)):
    """
    Fit a simple ARIMA model on the training series and evaluate it on
    a single horizon window (e.g., the last test window).

    Parameters
    ----------
    train_series : array-like
        Historical sales values used to fit ARIMA.
    y_true_window : array-like
        True sales for the forecast horizon (e.g., last test window).
    order : tuple
        ARIMA order (p, d, q). Default = (5, 1, 0).

    Returns
    -------
    metrics : dict or None
        {
          "MAPE": ...,
          "RMSE": ...,
          "MASE": ...
        }
        or None if ARIMA is not available or fails.
    """
    if ARIMA is None:
        print("statsmodels is not installed. Skipping ARIMA baseline.")
        return None

    y_train = np.asarray(train_series, dtype="float32")
    y_true = np.asarray(y_true_window, dtype="float32")
    horizon = len(y_true)

    try:
        model = ARIMA(y_train, order=order).fit()

        forecast_start = len(y_train)
        forecast_end = forecast_start + horizon - 1

        y_pred = model.predict(start=forecast_start, end=forecast_end)
        y_pred = np.asarray(y_pred, dtype="float32")

        mape_val = safe_mape(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        mase_val = mase(y_true, y_pred, y_train)

        return {
            "MAPE": mape_val,
            "RMSE": rmse_val,
            "MASE": mase_val,
        }

    except Exception as e:
        print(f"ARIMA fitting failed: {e}")
        return None


# =====================================================================
# LEADERBOARD HELPER
# =====================================================================

def build_leaderboard(metrics_dict: dict):
    """
    Convert a dict of model_name → metrics dict into a tidy DataFrame.

    Example input:
        {
          "Naive": {"MAPE": 30, "RMSE": 100, "MASE": 1.1},
          "LSTM":  {"MAPE": 20, "RMSE": 80,  "MASE": 0.8},
        }

    Output: DataFrame with columns:
        model, MAPE, RMSE, MASE
    """
    rows = []
    for model_name, m in metrics_dict.items():
        row = {"model": model_name}
        row.update(m)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["model", "MAPE", "RMSE", "MASE"])

    df = pd.DataFrame(rows)
    return df[["model", "MAPE", "RMSE", "MASE"]]
